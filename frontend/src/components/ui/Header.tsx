import React from 'react';
import { User as UserIcon, Sun, Moon } from 'lucide-react';
import { useTheme } from '../../hooks/useTheme';

interface HeaderProps {
  currentUser: { fullName: string } | null;
  onProfileClick: () => void;
}

const Header: React.FC<HeaderProps> = ({ currentUser, onProfileClick }) => {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between p-4 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm shadow-md">
      <div className="flex items-center gap-2">
        <img src="/logo.png" alt="Okada" className="h-8 w-8" />
        <h1 className="text-xl font-bold text-gray-800 dark:text-white">Okada</h1>
      </div>
      <div className="flex items-center gap-4">
        <button onClick={toggleTheme} className="p-2 rounded-full text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700">
          {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
        </button>
        {currentUser ? (
          <button onClick={onProfileClick} className="flex items-center gap-2 text-gray-800 dark:text-white bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 px-4 py-2 rounded-lg transition-colors">
            <UserIcon size={18} />
            <span>{currentUser.fullName}</span>
          </button>
        ) : (
          <button onClick={onProfileClick} className="text-white bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition-colors">
            Login / Sign Up
          </button>
        )}
      </div>
    </header>
  );
};

export default Header;
