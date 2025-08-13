// src/App.jsx

import React, { useState } from 'react';
import ChatWindow from './components/ChatWindow';
import ChatInput from './components/ChatInput';
import './App.css';

const API_BASE_URL = '/api';

function App() {
    const [messages, setMessages] = useState([
        { type: 'bot', text: 'Bonjour ! Posez-moi une question ou envoyez un document PDF.' }
    ]);
    const [isLoading, setIsLoading] = useState(false);

    // Fonction pour ajouter un nouveau message à la liste
    const addMessage = (text, type) => {
        setMessages(prev => [...prev, { text, type }]);
    };

    // Fonction pour envoyer une question à l'API
    const handleSendMessage = async (question) => {
        addMessage(question, 'user');
        setIsLoading(true);

        try {
            const response = await fetch(`${API_BASE_URL}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Une erreur de serveur est survenue.');
            }
            
            // L'API Flask renvoie du texte brut
            const answer = await response.text();
            addMessage(answer, 'bot');

        } catch (error) {
            console.error("Erreur lors de la requête :", error);
            addMessage(`Erreur : ${error.message}`, 'error');
        } finally {
            setIsLoading(false);
        }
    };

    // Fonction pour envoyer un fichier à l'API
    const handleFileUpload = async (file) => {
        const formData = new FormData();
        formData.append('file', file);
        
        addMessage(`Téléversement du fichier : ${file.name}...`, 'info');
        setIsLoading(true);

        try {
            const response = await fetch(`${API_BASE_URL}/upload-document`, {
                method: 'POST',
                body: formData, // Pas de header 'Content-Type', le navigateur le gère
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Erreur lors du téléversement.');
            }
            
            addMessage(data.message, 'info');

        } catch (error) {
            console.error("Erreur lors de l'upload :", error);
            addMessage(`Erreur d'upload : ${error.message}`, 'error');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="app-container">
            <h1>Chatbot RAG</h1>
            <ChatWindow messages={messages} isLoading={isLoading} />
            <ChatInput
                onSendMessage={handleSendMessage}
                onFileUpload={handleFileUpload}
                isLoading={isLoading}
            />
        </div>
    );
}

export default App;