"use client";

import React, { useEffect, useState } from 'react';

interface Particle {
    id: number;
    left: number;
    size: number;
    duration: number;
    delay: number;
    opacity: number;
}

export const SnowEffect: React.FC = () => {
    const [particles, setParticles] = useState<Particle[]>([]);

    useEffect(() => {
        const items: Particle[] = [];
        const count = 60;

        for (let i = 0; i < count; i++) {
            items.push({
                id: i,
                left: Math.random() * 100,
                size: 2 + Math.random() * 8, // 2px ~ 10px circles
                duration: 15 + Math.random() * 25, // 15s ~ 40s (slow)
                delay: Math.random() * 15,
                opacity: 0.15 + Math.random() * 0.4,
            });
        }

        setParticles(items);
    }, []);

    return (
        <div className="fixed inset-0 pointer-events-none z-[60] overflow-hidden">
            {particles.map((p) => (
                <div
                    key={p.id}
                    className="absolute rounded-full bg-white"
                    style={{
                        left: `${p.left}%`,
                        width: `${p.size}px`,
                        height: `${p.size}px`,
                        opacity: p.opacity,
                        animation: `snowfall ${p.duration}s linear infinite`,
                        animationDelay: `${p.delay}s`,
                    }}
                />
            ))}
        </div>
    );
};

export default SnowEffect;
