// src/components/ChatWindow.jsx

import React, { useRef, useEffect } from 'react';

const TypingIndicator = () => (
    <div className="message bot">
        <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
    </div>
);

const ChatWindow = ({ messages, isLoading }) => {
    const endOfMessagesRef = useRef(null);

    const scrollToBottom = () => {
        endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isLoading]);

    return (
        <div className="chat-window">
            {messages.map((msg, index) => (
                <div key={index} className={`message ${msg.type}`}>
                    {msg.text}
                </div>
            ))}
            {isLoading && <TypingIndicator />}
            <div ref={endOfMessagesRef} />
        </div>
    );
};

export default ChatWindow;