"use client";
import { useRef, useEffect, useState } from "react";
import { useChat } from "../hooks/useChat";
import { Paperclip, Send, MessageSquare, X, Bot, User as UserIcon, FileText, CheckSquare, Square } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// --- TYPE DEFINITIONS ---
interface User {
  email: string;
  fullName: string;
}

interface ChatbotProps {
    currentUser: User | null;
    openAuthModal: () => void;
}

// --- DOCUMENT SELECTION SUB-COMPONENT ---
interface DocSelectionViewProps {
    docs: string[];
    onConfirm: (selectedDocs: string[]) => void;
    onSkip: () => void; // <-- NEW: Prop for the skip action
}
const DocumentSelectionView = ({ docs, onConfirm, onSkip }: DocSelectionViewProps) => {
    const [selected, setSelected] = useState<string[]>([]);
    const toggleSelection = (docName: string) => { setSelected(prev => prev.includes(docName) ? prev.filter(d => d !== docName) : [...prev, docName]); };
    
    return (
        <div className="p-6 flex flex-col h-full">
            <h3 className="font-semibold text-gray-800">Select Documents</h3>
            <p className="text-sm text-gray-500 mt-1 mb-4">
                Choose documents to use for this chat session, or continue without selecting any.
            </p>
            <div className="flex-grow overflow-y-auto space-y-2 pr-2">
                {docs.map(doc => (
                    <div 
                        key={doc}
                        onClick={() => toggleSelection(doc)}
                        className={`flex items-center gap-3 p-3 border rounded-lg cursor-pointer transition-all ${selected.includes(doc) ? 'border-blue-500 bg-blue-50' : 'border-gray-200'}`}
                    >
                        {selected.includes(doc) ? <CheckSquare size={20} className="text-blue-600"/> : <Square size={20} className="text-gray-400"/>}
                        <FileText size={20} className="text-gray-600"/>
                        <span className="flex-grow truncate text-sm font-medium">{doc}</span>
                    </div>
                ))}
            </div>
            {/* --- NEW: Button container for both actions --- */}
            <div className="mt-4 flex flex-col gap-2">
                <button
                    onClick={() => onConfirm(selected)}
                    className="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700 transition-colors disabled:bg-gray-400"
                    disabled={selected.length === 0}
                >
                    Start Chat with {selected.length} Document(s)
                </button>
                <button
                    onClick={onSkip}
                    className="w-full text-center text-sm text-gray-500 hover:text-black py-2"
                >
                    Continue without documents
                </button>
            </div>
        </div>
    );
};

// --- MAIN CHATBOT COMPONENT ---
const Chatbot = ({ currentUser, openAuthModal }: ChatbotProps) => {
  const { 
    messages, setMessages, input, setInput, handleSendMessage, isLoading, isIndexing, 
    availableDocs, isAwaitingDocSelection, setIsAwaitingDocSelection, 
    loadSelectedDocuments, handleFileUpload, fetchHistory // <-- NEWLY IMPORTED
  } = useChat(currentUser?.email || null);

  const [isOpen, setIsOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatBoxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Automatically open the chatbot window if the app determines we need to select documents
    if (isAwaitingDocSelection && !isOpen) {
      setIsOpen(true);
    }
  }, [isAwaitingDocSelection]);

  useEffect(() => {
    if (isOpen) {
        if (!currentUser) {
            setMessages([{ role: 'bot', content: "Welcome! Please use the 'Login / Sign Up' button to start our conversation."}]);
        } else if (currentUser && !isAwaitingDocSelection && messages.length === 0) {
            fetchHistory();
        }
    }
  }, [isOpen, currentUser, isAwaitingDocSelection]);


useEffect(() => {
    if (chatBoxRef.current) {
      // Use setTimeout to ensure the scroll happens after the DOM has fully updated
      setTimeout(() => {
        chatBoxRef.current!.scrollTop = chatBoxRef.current!.scrollHeight;
      }, 0);
    }
  }, [messages, isOpen]);

  // --- THIS IS THE FIX ---
  // This function now fetches the history AFTER loading the documents.
  const onConfirmDocSelection = async (selectedDocs: string[]) => {
      const success = await loadSelectedDocuments(selectedDocs);
      if (success) {
          setIsAwaitingDocSelection(false);
          // Instead of a generic message, now we fetch the real history.
          fetchHistory();
      }
  };

    // --- NEW: Function to handle skipping document selection ---
  const onSkipDocSelection = () => {
      setIsAwaitingDocSelection(false); // Hide the selection screen
      fetchHistory(); // Fetch history to start the general chat
  };

  const handleOpenChat = () => {
    if (!currentUser) {
        openAuthModal();
    } else {
        setIsOpen(prev => !prev);
    }
  };

  const onFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  return (
    <>
<AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.3 }}
            className="fixed bottom-24 right-5 w-[90vw] max-w-md h-[70vh] max-h-[600px] bg-white rounded-xl shadow-2xl flex flex-col z-50"
          >
            <div className="flex items-center justify-between p-3 border-b border-gray-200 bg-gray-50 rounded-t-xl">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gray-200 rounded-full flex items-center justify-center"><Bot size={22} className="text-gray-600"/></div>
                <div>
                  <h2 className="font-semibold text-base">Okada IntelliAgent</h2>
                  <div className="flex items-center gap-1.5"><div className="w-2 h-2 bg-green-500 rounded-full"></div><p className="text-xs text-gray-500">Online</p></div>
                </div>
              </div>
              <button onClick={() => setIsOpen(false)} className="text-gray-500 hover:text-gray-900" title="Close Chat"><X size={20} /></button>
            </div>
            
            {isAwaitingDocSelection ? (
              <DocumentSelectionView docs={availableDocs} onConfirm={onConfirmDocSelection} onSkip={onSkipDocSelection} />
            ) : (
              <>
                <div ref={chatBoxRef} className="flex-1 p-4 overflow-y-auto">
                    {messages.map((msg, index) => (
                        <div key={index} className={`flex items-end gap-2 my-3 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                            <div className={`w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center ${msg.role === 'bot' ? 'bg-gray-200' : 'bg-blue-100'}`}>
                                {msg.role === 'bot' ? <Bot size={18} className="text-gray-600"/> : <UserIcon size={18} className="text-blue-600"/>}
                            </div>
                            <div className={`py-2 px-4 rounded-2xl max-w-[80%] whitespace-pre-wrap ${msg.role === 'user' ? 'bg-blue-600 text-white rounded-br-none' : 'bg-gray-200 text-gray-900 rounded-bl-none'}`}>
                                {msg.content}
                            </div>
                        </div>
                    ))}
                    {isLoading && ( <div className="flex items-end gap-2 my-3 flex-row"><div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center bg-gray-200"><Bot size={18} className="text-gray-600"/></div><div className="py-2 px-4 rounded-2xl bg-gray-200"><div className="flex items-center gap-1"><span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></span><span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0.1s]"></span><span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0.2s]"></span></div></div></div> )}
                </div>

                <div className="p-4 border-t border-gray-200">
                  <form onSubmit={(e) => { e.preventDefault(); handleSendMessage(input); setInput(''); }} className="relative">
                    <input type="text" value={input} onChange={(e) => setInput(e.target.value)} placeholder={isIndexing ? "Processing documents..." : "Type your message..."} className="w-full py-2 pl-10 pr-12 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100" autoComplete="off" disabled={isLoading || isIndexing || !currentUser} />
                    <button type="button" onClick={() => fileInputRef.current?.click()} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-blue-600 disabled:cursor-not-allowed" disabled={isLoading || isIndexing || !currentUser}><Paperclip size={20} /></button>
                    <input ref={fileInputRef} type="file" onChange={onFileUpload} accept=".csv,.pdf" className="hidden" />
                    <button type="submit" className="absolute right-2 top-1/2 -translate-y-1/2 bg-blue-600 text-white rounded-full w-8 h-8 flex items-center justify-center hover:bg-blue-700 disabled:bg-gray-400" disabled={isLoading || !input || !currentUser}><Send size={16} /></button>
                  </form>
                </div>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      <button onClick={handleOpenChat} className="fixed bottom-5 right-5 bg-blue-600 text-white w-14 h-14 rounded-full flex items-center justify-center shadow-lg hover:scale-110 transition-transform z-40" title="Toggle Chat">
        <MessageSquare size={28} />
      </button>
    </>
  );
};

export default Chatbot;
