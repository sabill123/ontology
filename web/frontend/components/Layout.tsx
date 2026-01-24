"use client";

import React, { useState, useEffect } from 'react';
import { Menu, X, Database, Brain, Target, Home, FileText, Play, Sun, Moon } from 'lucide-react';
import { SnowEffect } from './SnowEffect';
import { useTheme } from '@/contexts/ThemeContext';

// 테마 토글 버튼
function ThemeToggle() {
    const { theme, toggleTheme } = useTheme();

    return (
        <button
            onClick={toggleTheme}
            className="relative flex items-center justify-center w-9 h-9 rounded-full
                       bg-white/10 hover:bg-white/20 dark:bg-white/5 dark:hover:bg-white/10
                       border border-white/20 dark:border-white/10
                       transition-all duration-300"
            aria-label={theme === 'light' ? '다크 모드로 전환' : '라이트 모드로 전환'}
        >
            <Sun
                className={`absolute w-4 h-4 text-amber-400 transition-all duration-300
                           ${theme === 'light' ? 'opacity-100 rotate-0 scale-100' : 'opacity-0 rotate-90 scale-0'}`}
            />
            <Moon
                className={`absolute w-4 h-4 text-blue-300 transition-all duration-300
                           ${theme === 'dark' ? 'opacity-100 rotate-0 scale-100' : 'opacity-0 -rotate-90 scale-0'}`}
            />
        </button>
    );
}

export type ViewType = 'overview' | 'data' | 'ontology' | 'insights' | 'actions' | 'logs';

interface LayoutProps {
    children: React.ReactNode;
    activeView: ViewType;
    onNavigate: (view: ViewType) => void;
    isProcessing?: boolean;
    runFullPipeline?: () => void;
}

export const Layout: React.FC<LayoutProps> = ({ children, activeView, onNavigate, isProcessing, runFullPipeline }) => {
    const [isScrolled, setIsScrolled] = useState(false);
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            setIsScrolled(window.scrollY > 50);
        };
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const navLinks: { id: ViewType; label: string; icon: any }[] = [
        { id: 'overview', label: '개요', icon: Home },
        { id: 'data', label: '데이터 사일로', icon: Database },
        { id: 'ontology', label: '온톨로지', icon: Brain },
        { id: 'insights', label: '인사이트', icon: Target },
        { id: 'actions', label: '거버넌스', icon: Target },
        { id: 'logs', label: '실시간 로그', icon: FileText },
    ];

    const { theme } = useTheme();

    return (
        <div className="min-h-screen flex flex-col text-gray-900 dark:text-white selection:bg-emerald-500/30 relative overflow-x-hidden bg-gray-50 dark:bg-transparent transition-colors duration-300">
            {/* Atmospheric Background - only in dark mode */}
            {theme === 'dark' && <div className="atmospheric-bg" />}

            {/* Snow Effect - only in dark mode */}
            {theme === 'dark' && <SnowEffect />}

            {/* Sticky Glass Navbar */}
            <nav
                className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500
                    border-b border-gray-200 dark:border-white/5
                    ${isScrolled || isMobileMenuOpen
                        ? 'bg-white/90 dark:bg-black/60 backdrop-blur-lg shadow-sm dark:shadow-none py-4'
                        : 'bg-white dark:bg-transparent py-5'}`}
            >
                <div className="max-w-[1920px] mx-auto px-6 md:px-12 flex items-center justify-between">

                    {/* Mobile Menu Button */}
                    <button
                        className="md:hidden text-gray-700 dark:text-white p-2 -ml-2 hover:bg-gray-100 dark:hover:bg-white/10 rounded-full transition-colors"
                        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                    >
                        {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
                    </button>

                    {/* Desktop Links */}
                    <div className="hidden md:flex space-x-8 text-xs font-semibold tracking-widest uppercase text-gray-500 dark:text-zinc-400">
                        {navLinks.map(link => (
                            <button
                                key={link.id}
                                className={`hover:text-gray-900 dark:hover:text-white transition-colors duration-300 flex items-center gap-2 group
                                    ${activeView === link.id ? 'text-gray-900 dark:text-white' : ''}`}
                                onClick={() => onNavigate(link.id)}
                            >
                                <link.icon size={14} className={`transition-colors ${activeView === link.id ? 'text-emerald-500 dark:text-emerald-400' : 'text-gray-400 dark:text-zinc-600 group-hover:text-emerald-500 dark:group-hover:text-emerald-400'}`} />
                                {link.label}
                            </button>
                        ))}
                    </div>

                    {/* Logo */}
                    <div
                        className="absolute left-1/2 transform -translate-x-1/2 cursor-pointer group"
                        onClick={() => onNavigate('overview')}
                    >
                        <h1 className="text-xl md:text-2xl font-black tracking-[0.2em] uppercase select-none">
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-600 to-cyan-600 dark:from-emerald-400 dark:to-cyan-400 group-hover:from-emerald-500 group-hover:to-cyan-500 dark:group-hover:from-emerald-300 dark:group-hover:to-cyan-300 transition-all">LETOLOGY</span>
                        </h1>
                    </div>

                    {/* Right Icons */}
                    <div className="flex items-center space-x-6">
                        {runFullPipeline && (
                            <button
                                onClick={runFullPipeline}
                                disabled={isProcessing}
                                className="hidden md:flex items-center gap-2 px-4 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white border border-emerald-500 rounded-full text-xs font-bold uppercase transition-all disabled:opacity-50 shadow-lg shadow-emerald-500/20 active:scale-95"
                            >
                                {isProcessing ? (
                                    <div className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                ) : (
                                    <Play size={12} fill="currentColor" />
                                )}
                                {isProcessing ? '처리중...' : '파이프라인 실행'}
                            </button>
                        )}

                        <div className="hidden md:flex items-center space-x-2 bg-gray-100 dark:bg-white/5 rounded-full px-3 py-1.5 border border-gray-200 dark:border-white/10 hover:bg-gray-200 dark:hover:bg-white/10 transition-all cursor-pointer group hover:border-emerald-500/30">
                            <div className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-yellow-500 animate-pulse' : 'bg-emerald-500'}`}></div>
                            <span className="text-xs text-gray-600 dark:text-zinc-400 group-hover:text-gray-900 dark:group-hover:text-white uppercase font-medium">{isProcessing ? '시스템 처리 중' : '시스템 온라인'}</span>
                        </div>

                        {/* Theme Toggle */}
                        <ThemeToggle />
                    </div>
                </div>

                {/* Mobile Menu Overlay */}
                {isMobileMenuOpen && (
                    <div className="absolute top-full left-0 w-full glass-nav border-t border-white/10 h-screen md:hidden flex flex-col p-8 space-y-6 animate-in slide-in-from-top-5 duration-300">
                        {navLinks.map(link => (
                            <button
                                key={link.id}
                                className="text-xl font-light text-left border-b border-white/5 pb-4 flex items-center gap-4 text-zinc-300 active:text-white"
                                onClick={() => {
                                    onNavigate(link.id);
                                    setIsMobileMenuOpen(false);
                                }}
                            >
                                <div className="p-2 rounded-lg bg-white/5">
                                    <link.icon size={20} className="text-emerald-400" />
                                </div>
                                {link.label}
                            </button>
                        ))}
                    </div>
                )}
            </nav>

            {/* Main Content (with padding for fixed nav) */}
            <main className="flex-grow pt-24 min-h-screen relative z-10 flex flex-col items-center">
                <div className="w-full max-w-[1600px] mx-auto px-6 md:px-8">
                    {children}
                </div>
            </main>

            {/* Footer */}
            <footer className="bg-white dark:bg-black/90 border-t border-gray-200 dark:border-white/10 pt-20 pb-10 px-6 md:px-12 relative z-10">
                <div className="max-w-[1920px] mx-auto">
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-20">

                        <div className="space-y-4">
                            <h3 className="text-xs font-bold uppercase tracking-widest text-gray-400 dark:text-zinc-500 mb-6">시스템 상태</h3>
                            <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-zinc-400">
                                <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
                                API 서버: 온라인
                            </div>
                            <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-zinc-400">
                                <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
                                온톨로지 엔진: 활성
                            </div>
                            <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-zinc-400">
                                <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
                                액션 에이전트: 대기
                            </div>
                        </div>

                        <div className="space-y-4">
                            <h3 className="text-xs font-bold uppercase tracking-widest text-gray-400 dark:text-zinc-500 mb-6">모듈</h3>
                            <ul className="space-y-2 text-sm text-gray-600 dark:text-zinc-400">
                                <li className="hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer transition-colors">데이터 통합</li>
                                <li className="hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer transition-colors">스키마 매칭</li>
                                <li className="hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer transition-colors">엔티티 해결</li>
                                <li className="hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer transition-colors">거버넌스</li>
                            </ul>
                        </div>

                        <div className="space-y-4">
                            <h3 className="text-xs font-bold uppercase tracking-widest text-gray-400 dark:text-zinc-500 mb-6">문서</h3>
                            <ul className="space-y-2 text-sm text-gray-600 dark:text-zinc-400">
                                <li className="hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer transition-colors">API 레퍼런스</li>
                                <li className="hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer transition-colors">사용자 가이드</li>
                                <li className="hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer transition-colors">아키텍처</li>
                                <li className="hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer transition-colors">릴리즈 노트</li>
                            </ul>
                        </div>

                        <div className="space-y-4">
                            <h3 className="text-xs font-bold uppercase tracking-widest text-gray-400 dark:text-zinc-500 mb-6">지원</h3>
                            <p className="text-sm text-gray-600 dark:text-zinc-400">엔터프라이즈 지원이 활성화되었습니다.</p>
                            <button className="text-xs font-bold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300 border border-emerald-500/30 px-4 py-2 rounded">지원 문의</button>
                        </div>
                    </div>

                    <div className="pt-8 border-t border-gray-200 dark:border-white/10 flex flex-col md:flex-row justify-between items-center">
                        <h1 className="text-[5vw] md:text-[3vw] font-black tracking-tighter leading-none text-gray-100 dark:text-white/5 select-none uppercase">Letology</h1>
                        <div className="text-xs text-gray-400 dark:text-zinc-600 mt-4 md:mt-0 flex items-center gap-4">
                            <span>© 2025 Letology Inc.</span>
                            <span>v2.0.0 (Premium)</span>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    );
};
