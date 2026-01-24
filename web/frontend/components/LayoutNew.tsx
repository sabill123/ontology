"use client";

import React, { useState, useEffect } from 'react';
import { Menu, X, Database, Brain, Target, Home, FileText, Play, BarChart3, Settings, Bell, Search } from 'lucide-react';
import { ThemeToggle } from './ui/ThemeToggle';
import { useTheme } from '@/contexts/ThemeContext';

export type ViewType = 'overview' | 'data' | 'ontology' | 'insights' | 'governance' | 'logs';

interface LayoutProps {
    children: React.ReactNode;
    activeView: ViewType;
    onNavigate: (view: ViewType) => void;
    isProcessing?: boolean;
    runFullPipeline?: () => void;
}

export function LayoutNew({ children, activeView, onNavigate, isProcessing, runFullPipeline }: LayoutProps) {
    const [isScrolled, setIsScrolled] = useState(false);
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const { theme } = useTheme();

    useEffect(() => {
        const handleScroll = () => {
            setIsScrolled(window.scrollY > 20);
        };
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const navLinks: { id: ViewType; label: string; icon: any }[] = [
        { id: 'overview', label: '개요', icon: Home },
        { id: 'data', label: '데이터', icon: Database },
        { id: 'ontology', label: '온톨로지', icon: Brain },
        { id: 'insights', label: '인사이트', icon: BarChart3 },
        { id: 'governance', label: '거버넌스', icon: Target },
        { id: 'logs', label: '로그', icon: FileText },
    ];

    return (
        <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-slate-900 text-gray-900 dark:text-white transition-colors duration-300">
            {/* Navbar */}
            <nav
                className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300
                    ${isScrolled || isMobileMenuOpen
                        ? 'bg-white/90 dark:bg-slate-900/90 backdrop-blur-lg shadow-sm border-b border-gray-200 dark:border-slate-800'
                        : 'bg-transparent'}
                `}
            >
                <div className="max-w-7xl mx-auto px-4 md:px-6">
                    <div className="flex items-center justify-between h-16">
                        {/* Left: Logo & Nav */}
                        <div className="flex items-center gap-8">
                            {/* Logo */}
                            <div
                                className="flex items-center gap-2 cursor-pointer group"
                                onClick={() => onNavigate('overview')}
                            >
                                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center">
                                    <Brain className="w-5 h-5 text-white" />
                                </div>
                                <span className="text-lg font-bold tracking-tight">
                                    <span className="text-emerald-600 dark:text-emerald-400">LET</span>
                                    <span className="text-gray-700 dark:text-gray-300">OLOGY</span>
                                </span>
                            </div>

                            {/* Desktop Nav Links */}
                            <div className="hidden md:flex items-center gap-1">
                                {navLinks.map(link => (
                                    <button
                                        key={link.id}
                                        className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all
                                            ${activeView === link.id
                                                ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300'
                                                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-slate-800 hover:text-gray-900 dark:hover:text-white'}
                                        `}
                                        onClick={() => onNavigate(link.id)}
                                    >
                                        <link.icon className="w-4 h-4" />
                                        {link.label}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Right: Actions */}
                        <div className="flex items-center gap-3">
                            {/* Search (Desktop) */}
                            <button className="hidden md:flex items-center gap-2 px-3 py-2 rounded-lg
                                             bg-gray-100 dark:bg-slate-800 text-gray-500 dark:text-gray-400
                                             hover:bg-gray-200 dark:hover:bg-slate-700 transition-colors">
                                <Search className="w-4 h-4" />
                                <span className="text-sm">검색...</span>
                                <span className="text-xs text-gray-400 dark:text-gray-500 bg-gray-200 dark:bg-slate-700 px-1.5 py-0.5 rounded">⌘K</span>
                            </button>

                            {/* Pipeline Run Button */}
                            {runFullPipeline && (
                                <button
                                    onClick={runFullPipeline}
                                    disabled={isProcessing}
                                    className="hidden md:flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
                                               bg-gradient-to-r from-emerald-500 to-cyan-500 text-white
                                               hover:from-emerald-600 hover:to-cyan-600
                                               disabled:opacity-50 disabled:cursor-not-allowed
                                               shadow-lg shadow-emerald-500/20 hover:shadow-emerald-500/30
                                               transition-all active:scale-95"
                                >
                                    {isProcessing ? (
                                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    ) : (
                                        <Play className="w-4 h-4" fill="currentColor" />
                                    )}
                                    {isProcessing ? '처리중...' : '파이프라인 실행'}
                                </button>
                            )}

                            {/* System Status */}
                            <div className="hidden md:flex items-center gap-2 px-3 py-2 rounded-lg
                                           bg-gray-100 dark:bg-slate-800 border border-gray-200 dark:border-slate-700">
                                <div className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-amber-500 animate-pulse' : 'bg-emerald-500'}`} />
                                <span className="text-xs text-gray-600 dark:text-gray-400">
                                    {isProcessing ? '처리 중' : '온라인'}
                                </span>
                            </div>

                            {/* Notifications */}
                            <button className="relative p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-slate-800 transition-colors">
                                <Bell className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                                <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
                            </button>

                            {/* Theme Toggle */}
                            <ThemeToggle />

                            {/* Mobile Menu Button */}
                            <button
                                className="md:hidden p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-slate-800 transition-colors"
                                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                            >
                                {isMobileMenuOpen ? (
                                    <X className="w-5 h-5 text-gray-700 dark:text-gray-300" />
                                ) : (
                                    <Menu className="w-5 h-5 text-gray-700 dark:text-gray-300" />
                                )}
                            </button>
                        </div>
                    </div>
                </div>

                {/* Mobile Menu */}
                {isMobileMenuOpen && (
                    <div className="md:hidden bg-white dark:bg-slate-900 border-t border-gray-200 dark:border-slate-800">
                        <div className="px-4 py-3 space-y-1">
                            {navLinks.map(link => (
                                <button
                                    key={link.id}
                                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-colors
                                        ${activeView === link.id
                                            ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300'
                                            : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-slate-800'}
                                    `}
                                    onClick={() => {
                                        onNavigate(link.id);
                                        setIsMobileMenuOpen(false);
                                    }}
                                >
                                    <link.icon className="w-5 h-5" />
                                    <span className="font-medium">{link.label}</span>
                                </button>
                            ))}
                            {runFullPipeline && (
                                <button
                                    onClick={() => {
                                        runFullPipeline();
                                        setIsMobileMenuOpen(false);
                                    }}
                                    disabled={isProcessing}
                                    className="w-full flex items-center justify-center gap-2 px-4 py-3 mt-2 rounded-lg
                                               bg-gradient-to-r from-emerald-500 to-cyan-500 text-white font-medium
                                               disabled:opacity-50"
                                >
                                    {isProcessing ? (
                                        <>
                                            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                            처리중...
                                        </>
                                    ) : (
                                        <>
                                            <Play className="w-4 h-4" fill="currentColor" />
                                            파이프라인 실행
                                        </>
                                    )}
                                </button>
                            )}
                        </div>
                    </div>
                )}
            </nav>

            {/* Main Content */}
            <main className="flex-grow pt-20 pb-8 min-h-screen">
                <div className="max-w-7xl mx-auto px-4 md:px-6">
                    {children}
                </div>
            </main>

            {/* Footer */}
            <footer className="bg-white dark:bg-slate-900 border-t border-gray-200 dark:border-slate-800 py-8">
                <div className="max-w-7xl mx-auto px-4 md:px-6">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
                        {/* 시스템 상태 */}
                        <div>
                            <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-3">시스템 상태</h3>
                            <div className="space-y-2 text-sm">
                                <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                                    <div className="w-2 h-2 rounded-full bg-emerald-500" />
                                    API 서버: 온라인
                                </div>
                                <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                                    <div className="w-2 h-2 rounded-full bg-emerald-500" />
                                    온톨로지 엔진: 활성
                                </div>
                                <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                                    <div className="w-2 h-2 rounded-full bg-amber-500" />
                                    에이전트: 대기
                                </div>
                            </div>
                        </div>

                        {/* 모듈 */}
                        <div>
                            <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-3">모듈</h3>
                            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                                <li className="hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer">데이터 통합</li>
                                <li className="hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer">스키마 매칭</li>
                                <li className="hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer">엔티티 해결</li>
                                <li className="hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer">거버넌스</li>
                            </ul>
                        </div>

                        {/* 문서 */}
                        <div>
                            <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-3">문서</h3>
                            <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                                <li className="hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer">API 레퍼런스</li>
                                <li className="hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer">사용자 가이드</li>
                                <li className="hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer">아키텍처</li>
                            </ul>
                        </div>

                        {/* 지원 */}
                        <div>
                            <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-3">지원</h3>
                            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">엔터프라이즈 지원 활성화됨</p>
                            <button className="text-xs font-medium px-3 py-1.5 rounded-lg border border-emerald-500 text-emerald-600 dark:text-emerald-400 hover:bg-emerald-50 dark:hover:bg-emerald-900/20">
                                지원 문의
                            </button>
                        </div>
                    </div>

                    {/* 하단 */}
                    <div className="mt-8 pt-6 border-t border-gray-200 dark:border-slate-800 flex flex-col md:flex-row justify-between items-center gap-4">
                        <div className="flex items-center gap-2">
                            <div className="w-6 h-6 rounded-md bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center">
                                <Brain className="w-4 h-4 text-white" />
                            </div>
                            <span className="text-sm font-semibold text-gray-400 dark:text-gray-500">LETOLOGY</span>
                        </div>
                        <div className="flex items-center gap-4 text-xs text-gray-400 dark:text-gray-500">
                            <span>© 2025 Letology Inc.</span>
                            <span>v2.0.0</span>
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    );
}

export default LayoutNew;
