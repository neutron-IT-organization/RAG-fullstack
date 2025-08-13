// src/components/ChatInput.jsx

import React, { useState, useRef } from 'react';

const ChatInput = ({ onSendMessage, onFileUpload, isLoading }) => {
    const [input, setInput] = useState('');
    const fileInputRef = useRef(null);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (input.trim() && !isLoading) {
            onSendMessage(input);
            setInput('');
        }
    };

    const handleUploadClick = () => {
        fileInputRef.current.click();
    };

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file && !isLoading) {
            onFileUpload(file);
        }
    };

    return (
        <form onSubmit={handleSubmit} className="chat-input-form">
            <button
                type="button"
                className="upload-btn"
                onClick={handleUploadClick}
                disabled={isLoading}
                title="Téléverser un PDF"
            >
                📎
            </button>
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                style={{ display: 'none' }}
                accept=".pdf"
                disabled={isLoading}
            />
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Posez votre question..."
                disabled={isLoading}
            />
            <button type="submit" disabled={isLoading || !input.trim()} title="Envoyer">
                ➤
            </button>
        </form>
    );
};

export default ChatInput;