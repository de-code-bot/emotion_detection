import * as tf from '@tensorflow/tfjs';

export interface FaceBox {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
}

// Simple but effective face detection using image analysis
export class SimpleFaceDetector {
  private minFaceSize: number = 50;
  private maxFaceSize: number = 300;
  
  async detectFaces(canvas: HTMLCanvasElement): Promise<FaceBox[]> {
    const ctx = canvas.getContext('2d');
    if (!ctx || canvas.width === 0 || canvas.height === 0) {
      return [];
    }

    try {
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const faces: FaceBox[] = [];
      
      // Use multiple detection strategies
      const skinFaces = await this.detectBySkinColor(imageData, canvas.width, canvas.height);
      const edgeFaces = await this.detectByEdges(imageData, canvas.width, canvas.height);
      const centerFace = this.detectCenterFace(canvas.width, canvas.height);
      
      // Combine results
      faces.push(...skinFaces);
      faces.push(...edgeFaces);
      
      // If no faces found, add center detection as fallback
      if (faces.length === 0) {
        faces.push(centerFace);
      }
      
      // Remove duplicates and return best faces
      return this.filterAndRankFaces(faces, canvas.width, canvas.height);
      
    } catch (error) {
      console.error('Face detection error:', error);
      // Return center face as ultimate fallback
      return [this.detectCenterFace(canvas.width, canvas.height)];
    }
  }

  private async detectBySkinColor(imageData: ImageData, width: number, height: number): Promise<FaceBox[]> {
    const data = imageData.data;
    const skinMap = new Uint8Array(width * height);
    
    // Create skin color map
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const index = (y * width + x) * 4;
        const r = data[index];
        const g = data[index + 1];
        const b = data[index + 2];
        
        if (this.isSkinColor(r, g, b)) {
          skinMap[y * width + x] = 255;
        }
      }
    }
    
    // Find connected components of skin pixels
    const components = this.findConnectedComponents(skinMap, width, height);
    const faces: FaceBox[] = [];
    
    for (const component of components) {
      if (component.pixels.length > 100) { // Minimum size threshold
        const bounds = this.getBoundingBox(component.pixels);
        
        if (this.isValidFaceShape(bounds, width, height)) {
          faces.push({
            x: bounds.minX,
            y: bounds.minY,
            width: bounds.maxX - bounds.minX,
            height: bounds.maxY - bounds.minY,
            confidence: Math.min(1, component.pixels.length / 1000)
          });
        }
      }
    }
    
    return faces;
  }

  private async detectByEdges(imageData: ImageData, width: number, height: number): Promise<FaceBox[]> {
    const data = imageData.data;
    const grayData = new Uint8Array(width * height);
    
    // Convert to grayscale
    for (let i = 0; i < data.length; i += 4) {
      const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
      grayData[i / 4] = gray;
    }
    
    // Apply simple edge detection
    const edges = this.detectEdges(grayData, width, height);
    
    // Look for rectangular regions with high edge density
    const faces: FaceBox[] = [];
    const stepSize = 20;
    
    for (let y = 0; y < height - this.minFaceSize; y += stepSize) {
      for (let x = 0; x < width - this.minFaceSize; x += stepSize) {
        for (let size = this.minFaceSize; size <= Math.min(this.maxFaceSize, Math.min(width - x, height - y)); size += 20) {
          const edgeDensity = this.calculateEdgeDensity(edges, x, y, size, size, width);
          
          if (edgeDensity > 0.1 && edgeDensity < 0.7) { // Face-like edge density
            const aspectRatio = 1.0; // Square for simplicity
            const faceHeight = size * 1.2; // Slightly taller
            
            if (y + faceHeight <= height) {
              faces.push({
                x: x,
                y: y,
                width: size,
                height: faceHeight,
                confidence: edgeDensity
              });
            }
          }
        }
      }
    }
    
    return faces;
  }

  private detectCenterFace(width: number, height: number): FaceBox {
    // Default face in center of image
    const faceSize = Math.min(width, height) * 0.4;
    return {
      x: (width - faceSize) / 2,
      y: (height - faceSize) / 2,
      width: faceSize,
      height: faceSize * 1.2,
      confidence: 0.3
    };
  }

  private isSkinColor(r: number, g: number, b: number): boolean {
    // Improved skin color detection
    const rg = r - g;
    const rb = r - b;
    const gb = g - b;
    
    return (
      r > 95 && g > 40 && b > 20 &&
      Math.max(r, g, b) - Math.min(r, g, b) > 15 &&
      Math.abs(rg) > 15 && rg > 0 && rb > 0
    ) || (
      r > 220 && g > 210 && b > 170 &&
      Math.abs(r - g) <= 15 && r >= b && g >= b
    );
  }

  private findConnectedComponents(skinMap: Uint8Array, width: number, height: number) {
    const visited = new Uint8Array(width * height);
    const components: { pixels: { x: number; y: number }[] }[] = [];
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const index = y * width + x;
        
        if (skinMap[index] === 255 && visited[index] === 0) {
          const component = { pixels: [] as { x: number; y: number }[] };
          this.floodFill(skinMap, visited, x, y, width, height, component.pixels);
          
          if (component.pixels.length > 50) {
            components.push(component);
          }
        }
      }
    }
    
    return components;
  }

  private floodFill(
    skinMap: Uint8Array,
    visited: Uint8Array,
    startX: number,
    startY: number,
    width: number,
    height: number,
    pixels: { x: number; y: number }[]
  ) {
    const stack = [{ x: startX, y: startY }];
    
    while (stack.length > 0) {
      const { x, y } = stack.pop()!;
      const index = y * width + x;
      
      if (x < 0 || x >= width || y < 0 || y >= height || visited[index] === 1 || skinMap[index] !== 255) {
        continue;
      }
      
      visited[index] = 1;
      pixels.push({ x, y });
      
      // Add neighbors
      stack.push({ x: x + 1, y });
      stack.push({ x: x - 1, y });
      stack.push({ x, y: y + 1 });
      stack.push({ x, y: y - 1 });
    }
  }

  private getBoundingBox(pixels: { x: number; y: number }[]) {
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    
    for (const pixel of pixels) {
      minX = Math.min(minX, pixel.x);
      maxX = Math.max(maxX, pixel.x);
      minY = Math.min(minY, pixel.y);
      maxY = Math.max(maxY, pixel.y);
    }
    
    return { minX, maxX, minY, maxY };
  }

  private isValidFaceShape(bounds: any, imageWidth: number, imageHeight: number): boolean {
    const width = bounds.maxX - bounds.minX;
    const height = bounds.maxY - bounds.minY;
    
    // Check size constraints
    if (width < this.minFaceSize || height < this.minFaceSize) return false;
    if (width > this.maxFaceSize || height > this.maxFaceSize) return false;
    
    // Check aspect ratio (faces are typically 0.7 to 1.5 ratio)
    const aspectRatio = height / width;
    if (aspectRatio < 0.7 || aspectRatio > 2.0) return false;
    
    // Check position (not too close to edges)
    const margin = 10;
    if (bounds.minX < margin || bounds.minY < margin) return false;
    if (bounds.maxX > imageWidth - margin || bounds.maxY > imageHeight - margin) return false;
    
    return true;
  }

  private detectEdges(grayData: Uint8Array, width: number, height: number): Uint8Array {
    const edges = new Uint8Array(width * height);
    
    // Simple Sobel edge detection
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const gx = 
          -grayData[(y - 1) * width + (x - 1)] + grayData[(y - 1) * width + (x + 1)] +
          -2 * grayData[y * width + (x - 1)] + 2 * grayData[y * width + (x + 1)] +
          -grayData[(y + 1) * width + (x - 1)] + grayData[(y + 1) * width + (x + 1)];
        
        const gy = 
          -grayData[(y - 1) * width + (x - 1)] - 2 * grayData[(y - 1) * width + x] - grayData[(y - 1) * width + (x + 1)] +
          grayData[(y + 1) * width + (x - 1)] + 2 * grayData[(y + 1) * width + x] + grayData[(y + 1) * width + (x + 1)];
        
        const magnitude = Math.sqrt(gx * gx + gy * gy);
        edges[y * width + x] = magnitude > 50 ? 255 : 0;
      }
    }
    
    return edges;
  }

  private calculateEdgeDensity(edges: Uint8Array, x: number, y: number, width: number, height: number, imageWidth: number): number {
    let edgeCount = 0;
    let totalPixels = 0;
    
    for (let dy = 0; dy < height; dy++) {
      for (let dx = 0; dx < width; dx++) {
        const px = x + dx;
        const py = y + dy;
        
        if (px < imageWidth && py < edges.length / imageWidth) {
          const index = py * imageWidth + px;
          if (edges[index] === 255) edgeCount++;
          totalPixels++;
        }
      }
    }
    
    return totalPixels > 0 ? edgeCount / totalPixels : 0;
  }

  private filterAndRankFaces(faces: FaceBox[], width: number, height: number): FaceBox[] {
    if (faces.length === 0) return faces;
    
    // Remove overlapping faces (keep the one with higher confidence)
    const filtered: FaceBox[] = [];
    
    for (const face of faces) {
      let shouldAdd = true;
      
      for (let i = 0; i < filtered.length; i++) {
        const existing = filtered[i];
        const overlap = this.calculateOverlap(face, existing);
        
        if (overlap > 0.3) { // 30% overlap threshold
          if (face.confidence > existing.confidence) {
            filtered[i] = face; // Replace with better face
          }
          shouldAdd = false;
          break;
        }
      }
      
      if (shouldAdd) {
        filtered.push(face);
      }
    }
    
    // Sort by confidence and return top 3
    return filtered
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);
  }

  private calculateOverlap(face1: FaceBox, face2: FaceBox): number {
    const x1 = Math.max(face1.x, face2.x);
    const y1 = Math.max(face1.y, face2.y);
    const x2 = Math.min(face1.x + face1.width, face2.x + face2.width);
    const y2 = Math.min(face1.y + face1.height, face2.y + face2.height);
    
    if (x2 <= x1 || y2 <= y1) return 0;
    
    const overlapArea = (x2 - x1) * (y2 - y1);
    const face1Area = face1.width * face1.height;
    const face2Area = face2.width * face2.height;
    const unionArea = face1Area + face2Area - overlapArea;
    
    return overlapArea / unionArea;
  }
}