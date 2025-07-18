import { useState, useEffect } from 'react';
import toast from 'react-hot-toast';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

interface Message {
  role: 'user' | 'bot';
  content: string;
}

export interface AppointmentDetails {
  title: string;
  location: string;
  datetime: string;
  duration: number;
  attendees: string;
  description?: string;
}

export const useChat = (userEmail: string | null) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isIndexing, setIsIndexing] = useState(false);
    
    const [availableDocs, setAvailableDocs] = useState<string[]>([]);
    const [isAwaitingDocSelection, setIsAwaitingDocSelection] = useState(false);
    const [appointmentToConfirm, setAppointmentToConfirm] = useState<AppointmentDetails | null>(null);
    
    useEffect(() => {
        if (userEmail) {
            const fetchUserDocs = async () => {
                try {
                    const docsResponse = await fetch(`${API_BASE_URL}/documents/list/${encodeURIComponent(userEmail)}`);
                    if (docsResponse.ok) {
                        const docsData = await docsResponse.json();
                        if (docsData.documents && docsData.documents.length > 0) {
                            setAvailableDocs(docsData.documents);
                            setIsAwaitingDocSelection(true);
                        }
                    }
                } catch (error) {
                    toast.error("Could not fetch user documents.");
                }
            };
            fetchUserDocs();
        } else {
            setMessages([]);
            setAvailableDocs([]);
            setIsAwaitingDocSelection(false);
            setAppointmentToConfirm(null);
        }
    }, [userEmail]);

    const pollIndexingStatus = () => {
        const intervalId = setInterval(async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/documents/status`);
                const data = await response.json();

                if (data.status === 'success' || data.status === 'error' || data.status === 'idle') {
                    clearInterval(intervalId);
                    toast.dismiss();
                    
                    if (data.status === 'success') {
                        toast.success(data.message || "Documents ready!");
                        setIsAwaitingDocSelection(false);
                        fetchHistory();
                    } else {
                        toast.error(data.message || "Failed to load documents.");
                    }
                    setIsIndexing(false);
                }
            } catch (error) {
                clearInterval(intervalId);
                toast.dismiss();
                toast.error("Could not get document status.");
                setIsIndexing(false);
            }
        }, 3000);
    };
    
    const fetchHistory = async () => {
        if (!userEmail) return;

        try {
            const historyResponse = await fetch(`${API_BASE_URL}/conversations/${encodeURIComponent(userEmail)}`);
            if (historyResponse.ok) {
                const historyData = await historyResponse.json();
                if (historyData.history && historyData.history.length > 0) {
                    const formattedHistory = historyData.history.map((msg: any) => ({
                        role: msg.role === 'assistant' ? 'bot' : 'user',
                        content: msg.content,
                    }));
                    setMessages(formattedHistory);
                } else {
                    setMessages([{ role: 'bot', content: `Hi! How can I help you today?` }]);
                }
            } else {
                 setMessages([{ role: 'bot', content: `Hi! I couldn't retrieve your past conversations.` }]);
            }
        } catch (error) {
            toast.error("Failed to load chat history.");
        }
    };

    const handleSendMessage = async (message: string) => {
        if (!message || !userEmail) return;
        
        const userMessage: Message = { role: 'user', content: message };
        setMessages((prev) => [...prev, userMessage]);
        setIsLoading(true);

        try {
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: userEmail,
                    message: message,
                    history: [...messages, userMessage].map(m => ({
                        role: m.role === 'bot' ? 'assistant' : 'user',
                        content: m.content
                    })),
                }),
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to get a response.');
            }
            const data = await response.json();
            if (data.answer) {
                setMessages((prev) => [...prev, { role: 'bot', content: data.answer }]);
            }
            if (data.appointment_details) {
                setAppointmentToConfirm(data.appointment_details);
            } else {
                setAppointmentToConfirm(null);
            }
        } catch (error) {
            setMessages((prev) => [...prev, { role: 'bot', content: `Error: ${(error as Error).message}` }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleConfirmAppointment = async () => {
        if (!appointmentToConfirm || !userEmail) return;

        setIsLoading(true);
        try {
            const response = await fetch(`${API_BASE_URL}/appointments/schedule`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: userEmail,
                    ...appointmentToConfirm,
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to schedule appointment.');
            }
            const data = await response.json();
            if (data.message) {
                setMessages((prev) => [...prev, { role: 'bot', content: data.message }]);
            }
        } catch (error) {
            toast.error(`Confirmation failed: ${(error as Error).message}`);
            throw error;
        } finally {
            setIsLoading(false);
        }
    };

    const handleCancelAppointment = () => {
        setAppointmentToConfirm(null);
        setMessages((prev) => [...prev, { role: 'bot', content: "Okay, I've cancelled the scheduling request." }]);
    };
    
    const loadSelectedDocuments = async (selectedDocs: string[]): Promise<boolean> => {
        if (!userEmail || selectedDocs.length === 0) {
            toast.error("Please select at least one document.");
            return false;
        }
        
        setIsIndexing(true);
        toast.loading("Loading selected documents... This may take a moment.");

        try {
            const response = await fetch(`${API_BASE_URL}/documents/load`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: userEmail, filenames: selectedDocs }),
            });
            
            if (!response.ok) throw new Error("Failed to start the document loading process.");
            
            pollIndexingStatus();
            return true;

        } catch (error) {
             toast.dismiss();
             toast.error(`Failed to load documents: ${(error as Error).message}`);
             setIsIndexing(false);
             return false;
        }
    };

    const handleFileUpload = async (file: File) => {
        if (!userEmail) {
            toast.error("Please log in to upload documents.");
            return;
        }
        const formData = new FormData();
        formData.append('file', file);
        formData.append('user_id', userEmail);

        setIsIndexing(true);
        toast.loading(`Uploading ${file.name}...`);
        
        try {
            const response = await fetch(`${API_BASE_URL}/documents/upload`, {
                method: 'POST',
                body: formData,
            });
            if (!response.ok) throw new Error("File upload failed.");
            
            pollIndexingStatus();
        } catch (error) {
            toast.dismiss();
            toast.error(`Upload failed: ${(error as Error).message}`);
            setIsIndexing(false);
        }
    };

    return { 
        messages, setMessages, input, setInput, handleSendMessage, isLoading, isIndexing,
        availableDocs, isAwaitingDocSelection, setIsAwaitingDocSelection,
        loadSelectedDocuments, handleFileUpload,
        fetchHistory,
        appointmentToConfirm,
        setAppointmentToConfirm,
        handleConfirmAppointment,
        handleCancelAppointment,
    };
};
