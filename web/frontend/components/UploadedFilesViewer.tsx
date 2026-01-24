"use client"

import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';

// Uploaded Files Viewer Component
export default function UploadedFilesViewer({ API_URL }: { API_URL: string }) {
    const [files, setFiles] = useState<any[]>([]);
    const [selectedFile, setSelectedFile] = useState<string | null>(null);
    const [fileContent, setFileContent] = useState<any>(null);

    useEffect(() => {
        fetchFiles();
    }, []);

    const fetchFiles = () => {
        fetch(`${API_URL}/phase1/files`)
            .then(res => res.json())
            .then(setFiles)
            .catch(() => setFiles([]));
    };

    const viewFile = (filename: string) => {
        fetch(`${API_URL}/phase1/file/${filename}`)
            .then(res => res.json())
            .then(data => {
                setSelectedFile(filename);
                setFileContent(data);
            })
            .catch(console.error);
    };

    const deleteFile = (filename: string) => {
        if (!confirm(`Delete ${filename}?`)) return;

        fetch(`${API_URL}/phase1/file/${filename}`, { method: 'DELETE' })
            .then(() => {
                fetchFiles();
                if (selectedFile === filename) {
                    setSelectedFile(null);
                    setFileContent(null);
                }
            });
    };

    if (files.length === 0) return null;

    return (
        <div className="space-y-4">
            <h3 className="text-xl font-bold">Uploaded Files</h3>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {/* File List */}
                <div className="space-y-2">
                    {files.map((file: any, i: number) => (
                        <div
                            key={i}
                            className={cn(
                                "p-3 rounded border cursor-pointer transition-colors",
                                selectedFile === file.name
                                    ? "bg-primary/20 border-primary"
                                    : "bg-zinc-900 border-zinc-800 hover:border-zinc-700"
                            )}
                            onClick={() => viewFile(file.name)}
                        >
                            <div className="flex justify-between items-start">
                                <div className="flex-1 min-w-0">
                                    <p className="font-medium text-white truncate">{file.name}</p>
                                    <p className="text-xs text-zinc-500">
                                        {(file.size / 1024).toFixed(1)} KB • {file.type.toUpperCase()}
                                    </p>
                                </div>
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        deleteFile(file.name);
                                    }}
                                    className="ml-2 px-2 py-1 text-xs bg-red-500/20 text-red-400 rounded hover:bg-red-500/30"
                                >
                                    Delete
                                </button>
                            </div>
                        </div>
                    ))}
                </div>

                {/* File Content Viewer */}
                <div className="lg:col-span-2">
                    {fileContent ? (
                        <Card className="bg-zinc-900 border-zinc-800">
                            <CardHeader>
                                <CardTitle className="text-sm">
                                    {fileContent.filename}
                                    <span className="ml-2 text-xs text-zinc-500">
                                        {fileContent.type === 'csv' && `${fileContent.total_rows} rows`}
                                    </span>
                                </CardTitle>
                            </CardHeader>
                            <CardContent>
                                {fileContent.type === 'csv' && (
                                    <div className="overflow-auto max-h-96">
                                        <table className="w-full text-xs">
                                            <thead className="bg-zinc-800 sticky top-0">
                                                <tr>
                                                    {fileContent.columns.map((col: string, i: number) => (
                                                        <th key={i} className="px-2 py-1 text-left font-medium">{col}</th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {fileContent.data.map((row: any, i: number) => (
                                                    <tr key={i} className="border-t border-zinc-800">
                                                        {fileContent.columns.map((col: string, j: number) => (
                                                            <td key={j} className="px-2 py-1 text-zinc-300">{row[col]}</td>
                                                        ))}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                                {fileContent.type === 'json' && (
                                    <pre className="text-xs bg-black p-4 rounded overflow-auto max-h-96">
                                        {JSON.stringify(fileContent.data, null, 2)}
                                    </pre>
                                )}
                                {fileContent.type === 'text' && (
                                    <pre className="text-xs bg-black p-4 rounded overflow-auto max-h-96">
                                        {fileContent.content}
                                    </pre>
                                )}
                            </CardContent>
                        </Card>
                    ) : (
                        <div className="h-full min-h-[200px] flex items-center justify-center bg-zinc-900/50 rounded border border-dashed border-zinc-800">
                            <p className="text-zinc-500">Select a file to view its content</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
