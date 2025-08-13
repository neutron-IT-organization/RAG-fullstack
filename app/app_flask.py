import os

os.environ["AWS_S3_ENDPOINT"] = "192.168.0.150:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "EdKvYHNxP5IhbeVjhmQb"
os.environ["AWS_SECRET_ACCESS_KEY"] = "NnLOf0hFDPUWQ6ez2JajPLD75mFsGqdO0LrwhlcM"

import sys
import io
import torch
from minio import Minio
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from typing import List
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules['pysqlite3']

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline

load_dotenv()

RAG_STATE = {}
app = Flask(__name__)

class NeighborRetriever(BaseRetriever):
    vectorstore: Chroma
    all_docs: List[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        best_docs = self.vectorstore.similarity_search(query, k=1)
        if not best_docs:
            return []

        best_doc_content = best_docs[0].page_content
        
        try:
            page_contents = [doc.page_content for doc in self.all_docs]
            best_doc_index = page_contents.index(best_doc_content)
        except ValueError:
            return best_docs

        start_index = max(0, best_doc_index - 1)
        end_index = min(len(self.all_docs), best_doc_index + 2)

        return self.all_docs[start_index:end_index]

def setup_rag_pipeline():
    global RAG_STATE
    print("[INFO] --- Démarrage : Chargement des modèles et du pipeline RAG ---")

    # Connexion MinIO
    print("[INFO] Connexion à MinIO...")
    try:
        minio_client = Minio(
            os.getenv("AWS_S3_ENDPOINT"),
            access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            secure=False
        )
        RAG_STATE["minio_client"] = minio_client
        print("[OK] Connexion à MinIO réussie.")
    except Exception as e:
        print(f"[ERREUR] Connexion à MinIO échouée : {e}")
        RAG_STATE["minio_client"] = None

    # Chemins locaux
    MODELS_MOUNT_PATH = "/app/models"
    LLM_LOCAL_PATH = os.path.join(MODELS_MOUNT_PATH, "Meta-Llama-3.2-3B-Instruct")
    EMBEDDING_LOCAL_PATH = os.path.join(MODELS_MOUNT_PATH, "all-MiniLM-L6-v2")

    # Chargement modèles
    print("[INFO] Chargement du modèle LLM...")
    nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        LLM_LOCAL_PATH,
        quantization_config=nf4_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )
    print("[OK] Modèle LLM chargé.")

    print("[INFO] Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_LOCAL_PATH, trust_remote_code=True)
    print("[OK] Tokenizer chargé.")

    print("[INFO] Chargement du modèle d'embedding...")
    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_LOCAL_PATH,
        model_kwargs={'device': 'cuda'}
    )
    print("[OK] Modèle d'embedding chargé.")

    # Chargement documents depuis MinIO avec logs détaillés
    all_documents = []
    if RAG_STATE["minio_client"]:
        print("[INFO] Début du téléchargement des PDF depuis MinIO...")
        try:
            PDF_DIRECTORY_ON_MINIO = "documents/"
            LOCAL_PDF_DOWNLOAD_DIR = "/app/pdf_downloads/"
            os.makedirs(LOCAL_PDF_DOWNLOAD_DIR, exist_ok=True)
            
            print(f"[DEBUG] Répertoire MinIO : {PDF_DIRECTORY_ON_MINIO}")
            print(f"[DEBUG] Répertoire local de téléchargement : {LOCAL_PDF_DOWNLOAD_DIR}")

            pdf_objects = minio_client.list_objects(
                "reda-rag", 
                prefix=PDF_DIRECTORY_ON_MINIO, 
                recursive=True
            )

            found_files = False
            for obj in pdf_objects:
                print(f"[DEBUG] Fichier trouvé dans MinIO : {obj.object_name}")
                if obj.object_name.lower().endswith('.pdf'):
                    found_files = True
                    local_pdf_path = os.path.join(LOCAL_PDF_DOWNLOAD_DIR, os.path.basename(obj.object_name))
                    print(f"[INFO] Téléchargement {obj.object_name} vers {local_pdf_path}")
                    
                    try:
                        minio_client.fget_object("reda-rag", obj.object_name, local_pdf_path)
                        print(f"[OK] Fichier téléchargé : {local_pdf_path}")
                    except Exception as e:
                        print(f"[ERREUR] Impossible de télécharger {obj.object_name} : {e}")
                        continue

                    try:
                        loader = PyPDFLoader(local_pdf_path)
                        pages = loader.load()
                        print(f"[OK] {len(pages)} pages extraites depuis {local_pdf_path}")
                        all_documents.extend(pages)
                    except Exception as e:
                        print(f"[ERREUR] Impossible de lire le PDF {local_pdf_path} : {e}")
            
            if not found_files:
                print("[WARN] Aucun fichier PDF trouvé dans MinIO à ce préfixe.")

            print(f"[OK] Total : {len(all_documents)} pages chargées depuis MinIO.")
        
        except Exception as e:
            print(f"[ERREUR] Problème global lors du chargement depuis MinIO : {e}")
    else:
        print("[ERREUR] Aucun client MinIO disponible dans RAG_STATE.")

    # Split documents
    print("[INFO] Découpage des documents en chunks sémantiques...")
    text_splitter = SemanticChunker(embeddings)
    all_chunks = text_splitter.split_documents(all_documents)
    print(f"[OK] {len(all_chunks)} chunks créés.")

    # Création vector store
    print("[INFO] Initialisation du vector store...")
    vectorstore = Chroma.from_documents(documents=all_chunks, embedding=embeddings)
    print("[OK] Vector store prêt.")

    # Création retriever
    print("[INFO] Création du retriever personnalisé...")
    retriever = NeighborRetriever(vectorstore=vectorstore, all_docs=all_chunks)
    print("[OK] Retriever prêt.")

    # Création pipeline text generation
    print("[INFO] Initialisation du pipeline de génération de texte...")
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=4096,
        return_full_text=False
    )
    print("[OK] Pipeline de génération prêt.")

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    prompt_template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Tu es un assistant utile et bienveillant. Ton objectif est de fournir des réponses complètes et précises.

    Pour répondre à la question de l'utilisateur, réfère-toi d'abord au CONTEXTE FOURNI.
    Si la réponse NE SE TROUVE PAS dans le contexte, utilise alors tes propres connaissances pour y répondre.
    Si tu ne connais pas la réponse, qu'elle soit dans le contexte ou non, dis simplement que tu ne sais pas.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Contexte:
    {context}

    Question: {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    prompt = ChatPromptTemplate.from_template(prompt_template_str)

    print("[INFO] Création de la chain RAG...")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("[OK] Chaîne RAG créée.")

    # Stockage dans l'état global
    RAG_STATE.update({
        "model": model,
        "tokenizer": tokenizer,
        "retriever": retriever,
        "retrieval_chain": retrieval_chain,
        "text_splitter": text_splitter
    })

    print("Pipeline RAG initialisé avec succès.")

def transform_query_with_llm(question: str) -> str:
    transformation_prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Tu es un outil de reformulation de requêtes. Ton seul rôle est de réécrire la question de l'utilisateur pour la rendre optimale pour une recherche sémantique.
    Règles strictes:
    1. Ne réponds JAMAIS à la question.
    2. Garde le sens original de la question.
    3. Ta sortie doit être une et une seule question.
    4. La question reformulée doit être concise.
    
    Exemple 1:
    Utilisateur: infos sur la sécurité openshift
    Assistant: Quelles sont les meilleures pratiques de sécurité pour un cluster OpenShift ?
    
    Exemple 2:
    Utilisateur: tu peux me faire un résumé ?
    Assistant: Quel est le résumé du document fourni ?
    
    Exemple 3:
    Utilisateur: c'est quoi les grands titre dont tu peux m'aider, en se basant sur le contexte?
    Assistant: Quels sont les thèmes principaux abordés dans le document ?<|eot_id|><|start_header_id|>user<|end_header_id|>
    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    formatted_prompt = transformation_prompt_template.format(question=question)
    model = RAG_STATE["model"]
    tokenizer = RAG_STATE["tokenizer"]
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id)
    transformed_question = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return transformed_question.strip()

@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "La question est manquante"}), 400
    
    question = data["question"]
    print(f"Question originale reçue: \"{question}\"")
    
    transformed_question = transform_query_with_llm(question)
    print(f"Question transformée: \"{transformed_question}\"")
    
    response = RAG_STATE["retrieval_chain"].invoke({"input": transformed_question})
    
    return response["answer"]+"\n"

@app.route("/upload-document", methods=["POST"])
def upload_document():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier"}), 400
    
    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Fichier non PDF"}), 400

    minio_client = RAG_STATE.get("minio_client")
    if not minio_client:
        return jsonify({"error": "Connexion MinIO non dispo"}), 503

    try:
        temp_dir = "./temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, file.filename)
        file.save(temp_file_path)
        
        minio_client.fput_object("reda-rag", f"documents/{file.filename}", temp_file_path)
        
        loader = PyPDFLoader(temp_file_path)
        new_docs = loader.load()
        
        text_splitter = RAG_STATE["text_splitter"]
        new_chunks = text_splitter.split_documents(new_docs)

        retriever = RAG_STATE["retriever"]
        retriever.vectorstore.add_documents(new_chunks)
        retriever.all_docs.extend(new_chunks)
        
        print(f"Document '{file.filename}' ajouté au RAG ({len(new_chunks)} chunks).")
        os.remove(temp_file_path)
        
        return jsonify({"status": "success", "message": f"Document '{file.filename}' ajouté."})
    except Exception as e:
        print(f"ERREUR lors de l'upload : {e}")
        return jsonify({"error": f"Une erreur est survenue: {e}"}), 500

setup_rag_pipeline()