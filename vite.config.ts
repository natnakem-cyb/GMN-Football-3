import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import http from 'http';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

function autoBridgePlugin() {
  return {
    name: 'auto-bridge-server',
    configureServer() {
      // Check if bridge server is responding on 5050, if not, spawn it
      const checkReq = http.get('http://127.0.0.1:5050/health', (res) => {
        // Already active
      });
      checkReq.on('error', () => {
        console.log('[Vite] Launching background GMN Bridge Service on port 5050...');
        const child = spawn('npx', ['tsx', 'training/bridge_server.ts'], {
          env: { ...process.env, GMN_BRIDGE_PORT: '5050' },
          stdio: 'inherit',
          detached: false,
        });
        child.unref();
      });
    },
  };
}

function onnxWasmPlugin() {
  const onnxDist = path.resolve(__dirname, 'node_modules/onnxruntime-web/dist');
  return {
    name: 'onnx-wasm-plugin',
    configureServer(server: any) {
      server.middlewares.use((req: any, res: any, next: any) => {
        if (req.url && (req.url.startsWith('/onnx/') || req.url.startsWith('/ort-wasm/'))) {
          const fileName = req.url.replace(/^\/(onnx|ort-wasm)\//, '').split('?')[0];
          const filePath = path.join(onnxDist, fileName);
          if (fs.existsSync(filePath)) {
            if (fileName.endsWith('.wasm')) {
              res.setHeader('Content-Type', 'application/wasm');
            } else if (fileName.endsWith('.mjs') || fileName.endsWith('.js')) {
              res.setHeader('Content-Type', 'application/javascript');
            }
            res.setHeader('Access-Control-Allow-Origin', '*');
            const stream = fs.createReadStream(filePath);
            return stream.pipe(res);
          }
        }
        next();
      });
    },
    closeBundle() {
      const targetDir = path.resolve(__dirname, 'dist/onnx');
      if (fs.existsSync(onnxDist)) {
        if (!fs.existsSync(targetDir)) {
          fs.mkdirSync(targetDir, { recursive: true });
        }
        const files = fs.readdirSync(onnxDist);
        for (const file of files) {
          if (file.endsWith('.wasm') || file.endsWith('.mjs')) {
            fs.copyFileSync(path.join(onnxDist, file), path.join(targetDir, file));
          }
        }
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), autoBridgePlugin(), onnxWasmPlugin()],
  define: {
    'process.env': {},
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5050',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://127.0.0.1:5050',
        ws: true,
      },
    },
  },
});


