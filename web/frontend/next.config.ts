import type { NextConfig } from "next";

const isElectronBuild = process.env.BUILD_TARGET === 'electron';

const nextConfig: NextConfig = {
  output: isElectronBuild ? 'export' : undefined,
  images: {
    unoptimized: isElectronBuild,
  },
  trailingSlash: isElectronBuild,
};

export default nextConfig;
