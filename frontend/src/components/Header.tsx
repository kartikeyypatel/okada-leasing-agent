import React from 'react';

// Define an interface for the component's props
interface HeaderProps {
    children?: React.ReactNode; // The '?' makes the children prop optional
}

const Header = ({ children }: HeaderProps) => {
  return (
    <header className="py-6 px-8 flex items-center justify-between">
      <h1 className="text-2xl font-semibold text-white">Okada & Company</h1>
      {/* This div will render the buttons passed from App.tsx */}
      <div>
        {children}
      </div>
    </header>
  );
};

export default Header;
