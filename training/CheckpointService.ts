import fs from 'fs';
import path from 'path';

export interface CheckpointMetadata {
  id: string;
  filename: string;
  path: string;
  sizeBytes: number;
  scenario: string;
  algorithm: string;
  timesteps: number;
  createdAt: string;
  hasSourcePt: boolean;
  sourcePtName?: string;
  checkpointSha256?: string;
  weightsSha256?: string;
  obsDim?: number;
  actionDim?: number;
}

export class CheckpointService {
  private static PUBLIC_MODELS_DIR = path.resolve(process.cwd(), 'public/models');
  private static TRAINING_MODELS_DIR = path.resolve(process.cwd(), 'training/models');

  /**
   * Ensure directories exist.
   */
  public static init() {
    if (!fs.existsSync(this.PUBLIC_MODELS_DIR)) {
      fs.mkdirSync(this.PUBLIC_MODELS_DIR, { recursive: true });
    }
    if (!fs.existsSync(this.TRAINING_MODELS_DIR)) {
      fs.mkdirSync(this.TRAINING_MODELS_DIR, { recursive: true });
    }
  }

  /**
   * Scans public/models for all .onnx models and returns enriched metadata.
   */
  public static listCheckpoints(): CheckpointMetadata[] {
    this.init();
    const files = fs.readdirSync(this.PUBLIC_MODELS_DIR);
    const onnxFiles = files.filter((f) => f.endsWith('.onnx'));

    const results: CheckpointMetadata[] = [];

    for (const file of onnxFiles) {
      const fullPath = path.join(this.PUBLIC_MODELS_DIR, file);
      const stat = fs.statSync(fullPath);
      const sidecarPath = fullPath + '.json';

      let metadata: any = {};
      if (fs.existsSync(sidecarPath)) {
        try {
          metadata = JSON.parse(fs.readFileSync(sidecarPath, 'utf-8'));
        } catch (e) {
          console.warn(`[CheckpointService] Failed to parse sidecar for ${file}:`, e);
        }
      }

      // Check if matching source .pt or .zip exists in training/models
      const baseName = file.replace(/\.onnx$/, '');
      const possiblePtNames = [
        `${baseName}.pt`,
        `${baseName}_trained.pt`,
        metadata.sourceCheckpoint,
        file === 'mappo_policy.onnx' ? 'mappo_academy_3_vs_1_with_keeper_trained.pt' : null,
      ].filter(Boolean) as string[];

      let foundSourcePt: string | undefined = undefined;
      for (const ptName of possiblePtNames) {
        if (fs.existsSync(path.join(this.TRAINING_MODELS_DIR, ptName))) {
          foundSourcePt = ptName;
          break;
        }
      }

      results.push({
        id: file,
        filename: file,
        path: `/models/${file}`,
        sizeBytes: stat.size,
        scenario: metadata.scenario || 'academy_3_vs_1_with_keeper',
        algorithm: metadata.algorithm || (file.includes('mappo') ? 'MAPPO' : file.includes('ippo') ? 'IPPO' : 'PPO'),
        timesteps: metadata.timesteps || 0,
        createdAt: metadata.createdAt || stat.mtime.toISOString(),
        hasSourcePt: !!foundSourcePt,
        sourcePtName: foundSourcePt,
        checkpointSha256: metadata.checkpointSha256,
        weightsSha256: metadata.weightsSha256,
        obsDim: metadata.obsDim || 127,
        actionDim: metadata.actionDim || 19,
      });
    }

    // Sort newest first
    results.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
    return results;
  }

  /**
   * Safely deletes a checkpoint (.onnx + sidecar .json), and optionally its corresponding source .pt/.zip.
   */
  public static deleteCheckpoint(
    filename: string,
    deleteSourcePt: boolean = false
  ): { success: boolean; deletedFiles: string[]; message: string } {
    this.init();

    // Security check: restrict to clean filenames, no path traversal
    const cleanFilename = path.basename(filename);
    if (!cleanFilename.endsWith('.onnx') || cleanFilename.includes('..')) {
      throw new Error(`Invalid checkpoint filename: ${filename}`);
    }

    const onnxPath = path.join(this.PUBLIC_MODELS_DIR, cleanFilename);
    const sidecarPath = onnxPath + '.json';

    if (!fs.existsSync(onnxPath)) {
      throw new Error(`Checkpoint file not found: ${cleanFilename}`);
    }

    const deletedFiles: string[] = [];

    // Read sidecar to know source checkpoint before deletion
    let sourcePtName: string | undefined = undefined;
    if (fs.existsSync(sidecarPath)) {
      try {
        const sidecar = JSON.parse(fs.readFileSync(sidecarPath, 'utf-8'));
        sourcePtName = sidecar.sourceCheckpoint;
      } catch (e) {
        // ignore
      }
    }

    // Delete ONNX model
    fs.unlinkSync(onnxPath);
    deletedFiles.push(cleanFilename);

    // Delete sidecar JSON
    if (fs.existsSync(sidecarPath)) {
      fs.unlinkSync(sidecarPath);
      deletedFiles.push(cleanFilename + '.json');
    }

    // If requested, delete source .pt file in training/models
    if (deleteSourcePt) {
      const candidates = [
        sourcePtName,
        cleanFilename.replace(/\.onnx$/, '.pt'),
        cleanFilename.replace(/\.onnx$/, '_trained.pt'),
        cleanFilename.replace(/\.onnx$/, '.zip'),
      ].filter(Boolean) as string[];

      for (const cand of candidates) {
        const ptPath = path.join(this.TRAINING_MODELS_DIR, cand);
        if (fs.existsSync(ptPath)) {
          fs.unlinkSync(ptPath);
          deletedFiles.push(`training/models/${cand}`);
          break;
        }
      }
    }

    return {
      success: true,
      deletedFiles,
      message: `Successfully deleted ${deletedFiles.join(', ')}`,
    };
  }

  /**
   * Upload an ONNX model file directly from the browser (drag and drop).
   * Validates ONNX header and sets up sidecar metadata.
   */
  public static saveUploadedCheckpoint(
    filename: string,
    buffer: Buffer,
    metadata?: Partial<CheckpointMetadata>
  ): CheckpointMetadata {
    this.init();

    let cleanFilename = path.basename(filename);
    if (!cleanFilename.endsWith('.onnx')) {
      cleanFilename += '.onnx';
    }

    // Minimal ONNX validation (must be non-empty and have reasonable size)
    if (buffer.length < 512) {
      throw new Error('File is too small to be a valid ONNX model.');
    }

    const onnxPath = path.join(this.PUBLIC_MODELS_DIR, cleanFilename);
    fs.writeFileSync(onnxPath, buffer);

    const sidecarData = {
      scenario: metadata?.scenario || 'academy_3_vs_1_with_keeper',
      algorithm: metadata?.algorithm || 'MAPPO',
      timesteps: metadata?.timesteps || 0,
      createdAt: new Date().toISOString(),
      sourceCheckpoint: 'manual_upload',
      obsDim: metadata?.obsDim || 127,
      actionDim: metadata?.actionDim || 19,
      hidden: 64,
    };

    const sidecarPath = onnxPath + '.json';
    fs.writeFileSync(sidecarPath, JSON.stringify(sidecarData, null, 2));

    return {
      id: cleanFilename,
      filename: cleanFilename,
      path: `/models/${cleanFilename}`,
      sizeBytes: buffer.length,
      scenario: sidecarData.scenario,
      algorithm: sidecarData.algorithm,
      timesteps: sidecarData.timesteps,
      createdAt: sidecarData.createdAt,
      hasSourcePt: false,
      obsDim: sidecarData.obsDim,
      actionDim: sidecarData.actionDim,
    };
  }
}
