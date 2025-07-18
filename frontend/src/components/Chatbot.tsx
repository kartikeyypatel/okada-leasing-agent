"use client";
import { useRef, useEffect, useState } from "react";
import { useChat } from "../hooks/useChat";
import { Paperclip, Send, MessageSquare, X, User as UserIcon, FileText, CheckSquare, Square } from "lucide-react";
import { motion, AnimatePresence, useAnimation, easeInOut, easeOut } from "framer-motion";
import AppointmentConfirmation from './AppointmentConfirmation';
import { PureMultimodalInput } from "./ui/multimodal-ai-chat-input";
import { ThemeSwitch } from "./ui/theme-switch-button";
import { AIInputField } from "./ui/ai-input";
import { StarBorder } from "./ui/star-border";
import { Bot } from 'lucide-react';
import { cn } from '../lib/utils';

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

// AnimatedChatIcon component
const AnimatedChatIcon = ({
  size = 32,
  className = '',
  onClick
}: { size?: number; className?: string; onClick?: () => void }) => {
  const [isHovered, setIsHovered] = useState(false);
  const [isIdle, setIsIdle] = useState(false);
  const controls = useAnimation();
  const idleTimeoutRef = useRef<number | null>(null);

  const iconVariants = {
    normal: { scale: 1, rotate: 0, y: 0 },
    hover: { scale: 1.15, rotate: [0, -5, 5, -3, 3, 0], y: -2 },
    idle: { scale: [1, 1.05, 1], y: [0, -3, 0] },
    click: { scale: [1, 0.95, 1.1, 1], rotate: [0, -10, 10, 0] }
  };
  const glowVariants = {
    normal: { opacity: 0, scale: 1 },
    hover: { opacity: 0.6, scale: 1.3 },
    idle: { opacity: [0, 0.4, 0], scale: [1, 1.2, 1] }
  };
  const pulseVariants = {
    normal: { scale: 1, opacity: 0 },
    pulse: { scale: [1, 1.8, 2.2], opacity: [0.8, 0.3, 0] }
  };
  const iconTransition = { duration: 0.6, ease: easeInOut };
  const idleTransition = { duration: 2, ease: easeInOut, repeat: Infinity };
  const pulseTransition = { duration: 1.5, ease: easeOut, repeat: Infinity, repeatDelay: 3 };

  useEffect(() => {
    const startIdleAnimation = () => {
      idleTimeoutRef.current = window.setTimeout(() => {
        if (!isHovered) {
          setIsIdle(true);
          controls.start('idle');
        }
      }, 3000);
    };
    const resetIdleTimer = () => {
      if (idleTimeoutRef.current) clearTimeout(idleTimeoutRef.current);
      setIsIdle(false);
      startIdleAnimation();
    };
    resetIdleTimer();
    return () => { if (idleTimeoutRef.current) clearTimeout(idleTimeoutRef.current); };
  }, [isHovered, controls]);

  const handleMouseEnter = () => { setIsHovered(true); setIsIdle(false); controls.start('hover'); };
  const handleMouseLeave = () => { setIsHovered(false); controls.start('normal'); };
  const handleClick = () => { controls.start('click'); onClick?.(); };

  return (
    <div className="relative">
      {/* Pulse rings */}
      <motion.div
        className="absolute inset-0 rounded-full border-2 border-blue-400"
        variants={pulseVariants}
        animate="pulse"
        transition={pulseTransition}
        style={{
          left: '50%',
          top: '50%',
          transform: 'translate(-50%, -50%)',
          width: size + 24,
          height: size + 24,
        }}
      />
      <motion.div
        className="absolute inset-0 rounded-full border border-blue-300"
        variants={pulseVariants}
        animate="pulse"
        transition={{ ...pulseTransition, delay: 0.3 }}
        style={{
          left: '50%',
          top: '50%',
          transform: 'translate(-50%, -50%)',
          width: size + 32,
          height: size + 32,
        }}
      />
      {/* Main button container */}
      <motion.div
        className={cn(
          'relative cursor-pointer select-none rounded-full transition-all duration-300',
          'bg-gradient-to-br from-blue-500 to-blue-600 hover:from-blue-400 hover:to-blue-500',
          'shadow-lg hover:shadow-xl shadow-blue-500/25 hover:shadow-blue-500/40',
          'border border-blue-400/30 hover:border-blue-300/50',
          'backdrop-blur-sm',
          className
        )}
        style={{ padding: size * 0.4 }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        whileTap={{ scale: 0.95 }}
      >
        {/* Background glow */}
        <motion.div
          className="absolute inset-0 rounded-full bg-gradient-to-br from-blue-300/50 to-blue-500/50 blur-md"
          variants={glowVariants}
          animate={isHovered ? 'hover' : isIdle ? 'idle' : 'normal'}
          transition={isIdle ? idleTransition : iconTransition}
        />
        {/* Icon */}
        <motion.div
          className="relative z-10 flex items-center justify-center text-white"
          variants={iconVariants}
          animate={controls}
          transition={iconTransition}
        >
          <Bot size={size} />
        </motion.div>
      </motion.div>
    </div>
  );
};

// --- MAIN CHATBOT COMPONENT ---
const Chatbot = ({ currentUser, openAuthModal }: ChatbotProps) => {
  const { 
    messages, setMessages, input, setInput, handleSendMessage, isLoading, isIndexing, 
    availableDocs, isAwaitingDocSelection, setIsAwaitingDocSelection, 
    loadSelectedDocuments, handleFileUpload, fetchHistory, 
    appointmentToConfirm, handleConfirmAppointment, handleCancelAppointment
  } = useChat(currentUser?.email || null);

  const [isOpen, setIsOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatBoxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
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
      setTimeout(() => {
        chatBoxRef.current!.scrollTop = chatBoxRef.current!.scrollHeight;
      }, 0);
    }
  }, [messages, isOpen]);

  const onConfirmDocSelection = async (selectedDocs: string[]) => {
      const success = await loadSelectedDocuments(selectedDocs);
      if (success) {
          setIsAwaitingDocSelection(false);
          fetchHistory();
      }
  };

  const onSkipDocSelection = () => {
      setIsAwaitingDocSelection(false);
      fetchHistory();
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

  const typicalQuestions = [
    "Lease terms?",
    "Schedule tour",
    "Required docs",
    "Utilities included?"
  ];

  const [attachments, setAttachments] = useState<File[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const chatId = "main-chat"; // or your actual chat/session id

  const handleStopGenerating = () => {
    setIsGenerating(false);
  };

  const handleQuickQuestion = (question: string) => {
    handleSendMessage(question);
  };

  return (
    <>
      {appointmentToConfirm && (
        <AppointmentConfirmation 
            appointment={appointmentToConfirm}
            onConfirm={handleConfirmAppointment}
            onCancel={handleCancelAppointment}
            isLoading={isLoading}
        />
      )}

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.3 }}
            className="fixed bottom-24 right-5 w-[90vw] max-w-md h-[70vh] max-h-[600px] bg-white dark:bg-gray-900 rounded-xl shadow-2xl flex flex-col z-50 text-black dark:text-white"
          >
            <div className="flex items-center justify-between p-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 rounded-t-xl">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gray-200 dark:bg-gray-700 rounded-full flex items-center justify-center"><Bot size={22} className="text-gray-600 dark:text-gray-200"/></div>
                <div>
                  <h2 className="font-semibold text-base">Okada IntelliAgent</h2>
                  <div className="flex items-center gap-1.5"><div className="w-2 h-2 bg-green-500 rounded-full"></div><p className="text-xs text-gray-500 dark:text-gray-300">Online</p></div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <ThemeSwitch className="mr-1" />
                <button onClick={() => setIsOpen(false)} className="text-gray-500 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white" title="Close Chat"><X size={20} /></button>
              </div>
            </div>
            
            {isAwaitingDocSelection ? (
              <DocumentSelectionView docs={availableDocs} onConfirm={onConfirmDocSelection} onSkip={onSkipDocSelection} />
            ) : (
              <>
                <div ref={chatBoxRef} className="flex-1 p-4 overflow-y-auto">
                    {messages.map((msg, index) => (
                        <div key={index} className={`flex items-end gap-2 my-3 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                            <div className={`w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center ${msg.role === 'bot' ? 'bg-gray-200 dark:bg-gray-700' : 'bg-blue-100 dark:bg-blue-900'}`}>
                                {msg.role === 'bot' ? <Bot size={18} className="text-gray-600 dark:text-gray-200"/> : <UserIcon size={18} className="text-blue-600 dark:text-blue-300"/>}
                            </div>
                            <div className={`py-2 px-4 rounded-2xl max-w-[80%] whitespace-pre-wrap ${msg.role === 'user' ? 'bg-blue-600 dark:bg-blue-900 text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-white'} ${msg.role === 'user' ? 'rounded-br-none' : 'rounded-bl-none'}`}>
                                {msg.content}
                            </div>
                        </div>
                    ))}
                    {isLoading && ( <div className="flex items-end gap-2 my-3 flex-row"><div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center bg-gray-200 dark:bg-gray-700"><Bot size={18} className="text-gray-600 dark:text-gray-200"/></div><div className="py-2 px-4 rounded-2xl bg-gray-200 dark:bg-gray-800"><div className="flex items-center gap-1"><span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></span><span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0.1s]"></span><span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0.2s]"></span></div></div></div> )}
                </div>

                <div className="p-4 border-t border-gray-200">
                  {/* Quick actions */}
                  <div className="grid grid-cols-2 gap-1 mb-2">
                    {typicalQuestions.map((q, i) => (
                      <StarBorder
                        as="button"
                        key={i}
                        className="w-full h-8 px-1 py-0.5 text-[11px] font-medium truncate focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 transition-colors"
                        onClick={() => handleQuickQuestion(q)}
                        tabIndex={0}
                      >
                        {q}
                      </StarBorder>
                    ))}
                  </div>
                  {/* Animated AI input */}
                  <AIInputField 
                    onSendMessage={handleSendMessage}
                    onFileUpload={(files) => files.forEach(handleFileUpload)}
                    isLoading={isLoading}
                  />
                </div>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      <div className="fixed bottom-5 right-5 z-40">
        <AnimatedChatIcon size={40} onClick={handleOpenChat} />
      </div>
    </>
  );
};

export default Chatbot;
