import React from 'react';
import Chatbot from '../components/ui/Chatbot';

interface ChatPageProps {
  currentUser: { fullName: string } | null;
  openAuthModal: () => void;
}

const ChatPage: React.FC<ChatPageProps> = ({ currentUser, openAuthModal }) => {
  return (
    <div className="w-full h-full flex items-center justify-center">
      <Chatbot currentUser={currentUser} openAuthModal={openAuthModal} />
    </div>
  );
};

export default ChatPage; 