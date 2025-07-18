import React from 'react';

const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div className="relative min-h-screen w-full">
      <div className="absolute inset-0 bg-cover bg-center filter blur-sm" style={{ backgroundImage: "url('/nyc-background.jpg')" }}></div>
      <div className="absolute inset-0 bg-black/40"></div>
      
      <div className="relative z-10 flex flex-col min-h-screen">
        <main className="flex-grow flex items-center justify-center">
          {children}
        </main>
      </div>
    </div>
  );
};

export default MainLayout; 