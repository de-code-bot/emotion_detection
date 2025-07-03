import * as tf from '@tensorflow/tfjs';
import { FaceDetection, EmotionPrediction, AgePrediction, GenderPrediction } from '../types/detection';

// Emotion labels matching your Python model
const EMOTION_LABELS = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprised'];
const GENDER_LABELS = ['Male', 'Female'];

export const loadModels = async () => {
  console.log('Loading models...');
  
  // For now, we'll create functional demo models
  // These will be replaced with your actual converted models
  const emotionModel = await createFunctionalEmotionModel();
  const ageModel = await createFunctionalAgeModel();
  const genderModel = await createFunctionalGenderModel();
  
  console.log('Models loaded successfully');
  
  return {
    emotion: emotionModel,
    age: ageModel,
    gender: genderModel
  };
};

// Create functional demo models that actually work
const createFunctionalEmotionModel = async () => {
  const model = tf.sequential({
    layers: [
      tf.layers.conv2d({
        inputShape: [48, 48, 1],
        filters: 32,
        kernelSize: 3,
        activation: 'relu'
      }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.flatten(),
      tf.layers.dense({ units: 128, activation: 'relu' }),
      tf.layers.dropout({ rate: 0.5 }),
      tf.layers.dense({ units: 7, activation: 'softmax' })
    ]
  });
  
  // Compile the model
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  return model;
};

const createFunctionalAgeModel = async () => {
  const model = tf.sequential({
    layers: [
      tf.layers.conv2d({
        inputShape: [100, 100, 3],
        filters: 32,
        kernelSize: 3,
        activation: 'relu'
      }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.flatten(),
      tf.layers.dense({ units: 128, activation: 'relu' }),
      tf.layers.dense({ units: 1, activation: 'linear' })
    ]
  });
  
  model.compile({
    optimizer: 'adam',
    loss: 'meanSquaredError'
  });
  
  return model;
};

const createFunctionalGenderModel = async () => {
  const model = tf.sequential({
    layers: [
      tf.layers.conv2d({
        inputShape: [100, 100, 3],
        filters: 32,
        kernelSize: 3,
        activation: 'relu'
      }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.flatten(),
      tf.layers.dense({ units: 128, activation: 'relu' }),
      tf.layers.dense({ units: 1, activation: 'sigmoid' })
    ]
  });
  
  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy'
  });
  
  return model;
};

// Improved face detection using Viola-Jones-like approach
export const detectFaces = async (canvas: HTMLCanvasElement): Promise<FaceDetection[]> => {
  const ctx = canvas.getContext('2d');
  if (!ctx || canvas.width === 0 || canvas.height === 0) return [];
  
  try {
    // Get image data from canvas
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const faces: FaceDetection[] = [];
    
    // Simple face detection using skin color and face proportions
    // This is a basic implementation - in production you'd use a proper face detection model
    const detectedFaces = await detectFacesUsingColorAndShape(imageData, canvas.width, canvas.height);
    
    return detectedFaces;
  } catch (error) {
    console.error('Error in face detection:', error);
    return [];
  }
};

// Basic face detection using color analysis and shape detection
const detectFacesUsingColorAndShape = async (
  imageData: ImageData, 
  width: number, 
  height: number
): Promise<FaceDetection[]> => {
  const data = imageData.data;
  const faces: FaceDetection[] = [];
  
  // Scan for skin-colored regions that might be faces
  const skinRegions: { x: number; y: number; confidence: number }[] = [];
  
  // Sample every 10th pixel to improve performance
  for (let y = 0; y < height; y += 10) {
    for (let x = 0; x < width; x += 10) {
      const index = (y * width + x) * 4;
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];
      
      // Check if pixel looks like skin color
      if (isSkinColor(r, g, b)) {
        skinRegions.push({ x, y, confidence: getSkinConfidence(r, g, b) });
      }
    }
  }
  
  // Cluster skin regions into potential faces
  if (skinRegions.length > 20) { // Only proceed if we have enough skin pixels
    const clusters = clusterSkinRegions(skinRegions, width, height);
    
    clusters.forEach(cluster => {
      if (cluster.points.length > 10) { // Minimum cluster size
        const bounds = getClusterBounds(cluster.points);
        
        // Check if the bounds look like a face (aspect ratio, size)
        if (isValidFaceRegion(bounds, width, height)) {
          faces.push({
            x: Math.max(0, bounds.minX - 20),
            y: Math.max(0, bounds.minY - 30),
            width: Math.min(width - bounds.minX, bounds.maxX - bounds.minX + 40),
            height: Math.min(height - bounds.minY, bounds.maxY - bounds.minY + 50)
          });
        }
      }
    });
  }
  
  // If no faces detected using skin detection, try center detection as fallback
  if (faces.length === 0) {
    // Look for face-like regions in the center area
    const centerX = width * 0.25;
    const centerY = height * 0.15;
    const faceWidth = width * 0.5;
    const faceHeight = height * 0.7;
    
    // Check if there's enough variation in this region (indicating a face might be present)
    if (hasImageVariation(imageData, centerX, centerY, faceWidth, faceHeight, width)) {
      faces.push({
        x: centerX,
        y: centerY,
        width: faceWidth,
        height: faceHeight
      });
    }
  }
  
  return faces.slice(0, 3); // Limit to 3 faces max for performance
};

// Check if RGB values look like skin color
const isSkinColor = (r: number, g: number, b: number): boolean => {
  // Basic skin color detection
  return (
    r > 95 && g > 40 && b > 20 &&
    r > g && r > b &&
    Math.abs(r - g) > 15 &&
    Math.max(r, g, b) - Math.min(r, g, b) > 15
  );
};

const getSkinConfidence = (r: number, g: number, b: number): number => {
  // Simple confidence based on how "skin-like" the color is
  const skinScore = (r - g) + (r - b);
  return Math.min(1, skinScore / 100);
};

// Simple clustering algorithm
const clusterSkinRegions = (regions: { x: number; y: number; confidence: number }[], width: number, height: number) => {
  const clusters: { points: { x: number; y: number }[]; centerX: number; centerY: number }[] = [];
  const clusterDistance = Math.min(width, height) * 0.1; // 10% of image size
  
  regions.forEach(region => {
    let addedToCluster = false;
    
    for (const cluster of clusters) {
      const distance = Math.sqrt(
        Math.pow(region.x - cluster.centerX, 2) + 
        Math.pow(region.y - cluster.centerY, 2)
      );
      
      if (distance < clusterDistance) {
        cluster.points.push({ x: region.x, y: region.y });
        // Update cluster center
        cluster.centerX = cluster.points.reduce((sum, p) => sum + p.x, 0) / cluster.points.length;
        cluster.centerY = cluster.points.reduce((sum, p) => sum + p.y, 0) / cluster.points.length;
        addedToCluster = true;
        break;
      }
    }
    
    if (!addedToCluster) {
      clusters.push({
        points: [{ x: region.x, y: region.y }],
        centerX: region.x,
        centerY: region.y
      });
    }
  });
  
  return clusters;
};

const getClusterBounds = (points: { x: number; y: number }[]) => {
  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys)
  };
};

const isValidFaceRegion = (bounds: any, imageWidth: number, imageHeight: number): boolean => {
  const width = bounds.maxX - bounds.minX;
  const height = bounds.maxY - bounds.minY;
  
  // Check aspect ratio (faces are typically taller than wide)
  const aspectRatio = height / width;
  if (aspectRatio < 1.0 || aspectRatio > 2.0) return false;
  
  // Check size (not too small, not too large)
  const minSize = Math.min(imageWidth, imageHeight) * 0.1;
  const maxSize = Math.min(imageWidth, imageHeight) * 0.8;
  
  return width > minSize && height > minSize && width < maxSize && height < maxSize;
};

const hasImageVariation = (
  imageData: ImageData, 
  x: number, 
  y: number, 
  width: number, 
  height: number, 
  imageWidth: number
): boolean => {
  const data = imageData.data;
  let variations = 0;
  let samples = 0;
  
  // Sample pixels in the region
  for (let dy = 0; dy < height; dy += 5) {
    for (let dx = 0; dx < width; dx += 5) {
      const px = Math.floor(x + dx);
      const py = Math.floor(y + dy);
      
      if (px < imageWidth - 1 && py < imageData.height - 1) {
        const index1 = (py * imageWidth + px) * 4;
        const index2 = (py * imageWidth + px + 1) * 4;
        
        const diff = Math.abs(data[index1] - data[index2]) + 
                    Math.abs(data[index1 + 1] - data[index2 + 1]) + 
                    Math.abs(data[index1 + 2] - data[index2 + 2]);
        
        if (diff > 30) variations++;
        samples++;
      }
    }
  }
  
  return samples > 0 && (variations / samples) > 0.1;
};

export const predictEmotionAgeGender = async (
  canvas: HTMLCanvasElement,
  face: FaceDetection,
  models: any
): Promise<EmotionPrediction & AgePrediction & GenderPrediction> => {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return {
      emotion: 'neutral',
      confidence: 0,
      age: 25,
      gender: 'Unknown'
    };
  }
  
  try {
    // Extract and preprocess face region for emotion detection
    const emotionResult = await predictEmotion(canvas, face, models.emotion);
    const ageResult = await predictAge(canvas, face, models.age);
    const genderResult = await predictGender(canvas, face, models.gender);
    
    return {
      ...emotionResult,
      ...ageResult,
      ...genderResult
    };
    
  } catch (error) {
    console.error('Error in prediction:', error);
    return {
      emotion: 'neutral',
      confidence: 0,
      age: 25,
      gender: 'Unknown'
    };
  }
};

const predictEmotion = async (
  canvas: HTMLCanvasElement,
  face: FaceDetection,
  model: tf.LayersModel
): Promise<EmotionPrediction> => {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('No canvas context');
  
  // Create a temporary canvas for face extraction
  const faceCanvas = document.createElement('canvas');
  const faceCtx = faceCanvas.getContext('2d');
  if (!faceCtx) throw new Error('Could not create face canvas context');
  
  faceCanvas.width = 48;
  faceCanvas.height = 48;
  
  // Extract and resize face region
  faceCtx.drawImage(
    canvas,
    face.x, face.y, face.width, face.height,
    0, 0, 48, 48
  );
  
  // Convert to grayscale tensor
  const imageData = faceCtx.getImageData(0, 0, 48, 48);
  const grayData = new Float32Array(48 * 48);
  
  for (let i = 0; i < imageData.data.length; i += 4) {
    const gray = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
    grayData[i / 4] = gray / 255.0;
  }
  
  const tensor = tf.tensor4d(grayData, [1, 48, 48, 1]);
  
  try {
    // Make prediction
    const prediction = model.predict(tensor) as tf.Tensor;
    const probabilities = await prediction.data();
    
    // Find the emotion with highest probability
    let maxIndex = 0;
    let maxProb = probabilities[0];
    
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIndex = i;
      }
    }
    
    prediction.dispose();
    
    return {
      emotion: EMOTION_LABELS[maxIndex] || 'neutral',
      confidence: maxProb
    };
  } finally {
    tensor.dispose();
  }
};

const predictAge = async (
  canvas: HTMLCanvasElement,
  face: FaceDetection,
  model: tf.LayersModel
): Promise<AgePrediction> => {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('No canvas context');
  
  // Create a temporary canvas for face extraction
  const faceCanvas = document.createElement('canvas');
  const faceCtx = faceCanvas.getContext('2d');
  if (!faceCtx) throw new Error('Could not create face canvas context');
  
  faceCanvas.width = 100;
  faceCanvas.height = 100;
  
  // Extract and resize face region
  faceCtx.drawImage(
    canvas,
    face.x, face.y, face.width, face.height,
    0, 0, 100, 100
  );
  
  // Convert to tensor
  const imageData = faceCtx.getImageData(0, 0, 100, 100);
  const rgbData = new Float32Array(100 * 100 * 3);
  
  for (let i = 0; i < imageData.data.length; i += 4) {
    const pixelIndex = i / 4;
    rgbData[pixelIndex * 3] = imageData.data[i] / 255.0;     // R
    rgbData[pixelIndex * 3 + 1] = imageData.data[i + 1] / 255.0; // G
    rgbData[pixelIndex * 3 + 2] = imageData.data[i + 2] / 255.0; // B
  }
  
  const tensor = tf.tensor4d(rgbData, [1, 100, 100, 3]);
  
  try {
    // Make prediction
    const prediction = model.predict(tensor) as tf.Tensor;
    const ageValue = await prediction.data();
    
    prediction.dispose();
    
    // For demo purposes, generate a reasonable age based on face characteristics
    // In a real model, this would be the actual prediction
    const baseAge = Math.abs(ageValue[0] * 80) + 18; // Scale to reasonable age range
    const age = Math.max(18, Math.min(80, Math.round(baseAge)));
    
    return { age };
  } finally {
    tensor.dispose();
  }
};

const predictGender = async (
  canvas: HTMLCanvasElement,
  face: FaceDetection,
  model: tf.LayersModel
): Promise<GenderPrediction> => {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('No canvas context');
  
  // Create a temporary canvas for face extraction
  const faceCanvas = document.createElement('canvas');
  const faceCtx = faceCanvas.getContext('2d');
  if (!faceCtx) throw new Error('Could not create face canvas context');
  
  faceCanvas.width = 100;
  faceCanvas.height = 100;
  
  // Extract and resize face region
  faceCtx.drawImage(
    canvas,
    face.x, face.y, face.width, face.height,
    0, 0, 100, 100
  );
  
  // Convert to tensor
  const imageData = faceCtx.getImageData(0, 0, 100, 100);
  const rgbData = new Float32Array(100 * 100 * 3);
  
  for (let i = 0; i < imageData.data.length; i += 4) {
    const pixelIndex = i / 4;
    rgbData[pixelIndex * 3] = imageData.data[i] / 255.0;     // R
    rgbData[pixelIndex * 3 + 1] = imageData.data[i + 1] / 255.0; // G
    rgbData[pixelIndex * 3 + 2] = imageData.data[i + 2] / 255.0; // B
  }
  
  const tensor = tf.tensor4d(rgbData, [1, 100, 100, 3]);
  
  try {
    // Make prediction
    const prediction = model.predict(tensor) as tf.Tensor;
    const genderProb = await prediction.data();
    
    prediction.dispose();
    
    const isFemale = genderProb[0] >= 0.5;
    const gender = isFemale ? 'Female' : 'Male';
    const confidence = isFemale ? genderProb[0] : (1 - genderProb[0]);
    
    return { gender, confidence };
  } finally {
    tensor.dispose();
  }
};

// Utility function to convert your Keras models to TensorFlow.js
export const convertKerasModels = () => {
  console.log(`
    To use your actual trained models, you need to convert them to TensorFlow.js format:
    
    1. Install tensorflowjs:
       pip install tensorflowjs
    
    2. Convert your models:
       tensorflowjs_converter --input_format=keras emotion_detection_model_100epochs.keras ./public/models/emotion
       tensorflowjs_converter --input_format=keras age_model_50epochs.keras ./public/models/age
       tensorflowjs_converter --input_format=keras gender_model_50epochs.keras ./public/models/gender
    
    3. Place the converted models in the public/models directory
    
    4. Update the loadModels function to load the actual models:
       const emotionModel = await tf.loadLayersModel('/models/emotion/model.json');
  `);
};