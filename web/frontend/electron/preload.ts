const { contextBridge, ipcRenderer } = require('electron');

export interface ElectronAPI {
  readFile: (filePath: string) => Promise<{ success: boolean; content?: string; error?: string }>;
  showOpenDialog: (options: unknown) => Promise<unknown>;
  getAppInfo: () => Promise<{ version: string; name: string; platform: string; isDev: boolean }>;
  getSystemTheme: () => Promise<'light' | 'dark'>;
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
  onThemeChanged: (callback: (theme: 'light' | 'dark') => void) => () => void;
  onFolderSelected: (callback: (path: string) => void) => () => void;
  windowMinimize: () => void;
  windowMaximize: () => void;
  windowClose: () => void;
  platform: string;
  isElectron: boolean;
}

function createEventListener<T>(channel: string, callback: (data: T) => void): () => void {
  const listener = (_event: unknown, data: T) => callback(data);
  ipcRenderer.on(channel, listener);
  return () => ipcRenderer.removeListener(channel, listener);
}

const electronAPI: ElectronAPI = {
  readFile: (filePath: string) => ipcRenderer.invoke('read-file', filePath),
  showOpenDialog: (options: unknown) => ipcRenderer.invoke('show-open-dialog', options),
  getAppInfo: () => ipcRenderer.invoke('get-app-info'),
  getSystemTheme: () => ipcRenderer.invoke('get-system-theme'),
  setTheme: (theme: 'light' | 'dark' | 'system') => ipcRenderer.send('set-theme', theme),
  onThemeChanged: (callback: (theme: 'light' | 'dark') => void) => createEventListener('theme-changed', callback),
  onFolderSelected: (callback: (path: string) => void) => createEventListener('folder-selected', callback),
  windowMinimize: () => ipcRenderer.send('window-minimize'),
  windowMaximize: () => ipcRenderer.send('window-maximize'),
  windowClose: () => ipcRenderer.send('window-close'),
  platform: process.platform,
  isElectron: true,
};

contextBridge.exposeInMainWorld('electron', electronAPI);

declare global {
  interface Window {
    electron: ElectronAPI;
  }
}
