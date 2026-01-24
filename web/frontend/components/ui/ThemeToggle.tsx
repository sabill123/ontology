"use client";

import React from 'react';
import { Sun, Moon } from 'lucide-react';
import { useTheme } from '@/contexts/ThemeContext';

export function ThemeToggle() {
    const { theme, toggleTheme } = useTheme();

    return (
        <button
            onClick={toggleTheme}
            className="relative flex items-center justify-center w-10 h-10 rounded-full
                       bg-gray-100 dark:bg-slate-800
                       hover:bg-gray-200 dark:hover:bg-slate-700
                       border border-gray-200 dark:border-slate-700
                       transition-all duration-300 group"
            aria-label={theme === 'light' ? '다크 모드로 전환' : '라이트 모드로 전환'}
        >
            <Sun
                className={`absolute w-5 h-5 text-amber-500 transition-all duration-300
                           ${theme === 'light' ? 'opacity-100 rotate-0 scale-100' : 'opacity-0 rotate-90 scale-0'}`}
            />
            <Moon
                className={`absolute w-5 h-5 text-blue-400 transition-all duration-300
                           ${theme === 'dark' ? 'opacity-100 rotate-0 scale-100' : 'opacity-0 -rotate-90 scale-0'}`}
            />
        </button>
    );
}

export default ThemeToggle;
