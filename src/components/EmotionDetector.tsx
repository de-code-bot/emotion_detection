import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import { DetectionResult } from '../types/detection';
import { loadModels, detectFaces, predictEmotionAgeGender } from '../utils/modelUtils';
import WebcamView from './WebcamView';
import DetectionOverlay from './DetectionOverlay';
import ControlPanel from './ControlPanel';
import StatsPanel from './StatsPanel';

const EmotionDetector: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [detections, setDetections] = useState<DetectionResult[]>([]);
  const [models, setModels] = useState<any>(null);
  const [error, setError] = useState<string>('');
  const [fps, setFps] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const animationFrameRef = useRef<number>();
  const lastFrameTimeRef = useRef<number>(0);
  const processingTimeoutRef = useRef<NodeJS.Timeout>();

  // Initialize TensorFlow.js and load models
  useEffect(() => {
    const initializeTensorFlow = async () => {
      try {
        // Set backend to webgl for better performance
        await tf.setBackend('webgl');
        await tf.ready();
        console.log('TensorFlow.js initialized with WebGL backend');
        
        const loadedModels = await loadModels();
        setModels(loadedModels);
        setIsLoading(false);
        console.log('All models loaded successfully');
      } catch (err) {
        console.error('Failed to initialize TensorFlow.js:', err);
        setError('Failed to initialize AI models. Please refresh the page.');
        setIsLoading(false);
      }
    };

    initializeTensorFlow();
  }, []);

  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: { ideal: 640 }, 
          height: { ideal: 480 },
          facingMode: 'user'
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsWebcamActive(true);
        setError('');
        console.log('Webcam started successfully');
      }
    } catch (err) {
      console.error('Error accessing webcam:', err);
      setError('Unable to access webcam. Please ensure you have granted camera permissions.');
    }
  }, []);

  const stopWebcam = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsWebcamActive(false);
    setDetections([]);
    setIsProcessing(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (processingTimeoutRef.current) {
      clearTimeout(processingTimeoutRef.current);
    }
    console.log('Webcam stopped');
  }, []);

  const processFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !models || !isWebcamActive || isProcessing) {
      animationFrameRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx || video.videoWidth === 0 || video.videoHeight === 0) {
      animationFrameRef.current = requestAnimationFrame(processFrame);
      return;
    }

    // Calculate FPS
    const currentTime = performance.now();
    if (lastFrameTimeRef.current) {
      const deltaTime = currentTime - lastFrameTimeRef.current;
      setFps(Math.round(1000 / deltaTime));
    }
    lastFrameTimeRef.current = currentTime;

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Process detection every few frames to improve performance
    if (!isProcessing) {
      setIsProcessing(true);
      
      // Use setTimeout to prevent blocking the main thread
      processingTimeoutRef.current = setTimeout(async () => {
        try {
          console.log('Processing frame for face detection...');
          
          // Detect faces
          const faces = await detectFaces(canvas);
          console.log(`Detected ${faces.length} faces`);
          
          const results: DetectionResult[] = [];

          // Process each detected face
          for (const face of faces) {
            try {
              const prediction = await predictEmotionAgeGender(canvas, face, models);
              results.push({
                ...face,
                ...prediction
              });
              console.log('Face processed:', prediction);
            } catch (err) {
              console.error('Error processing individual face:', err);
            }
          }

          setDetections(results);
        } catch (err) {
          console.error('Error processing frame:', err);
        } finally {
          setIsProcessing(false);
        }
      }, 100); // Process every 100ms
    }

    animationFrameRef.current = requestAnimationFrame(processFrame);
  }, [models, isWebcamActive, isProcessing]);

  // Start processing when webcam is active
  useEffect(() => {
    if (isWebcamActive && models) {
      console.log('Starting frame processing...');
      processFrame();
    }
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (processingTimeoutRef.current) {
        clearTimeout(processingTimeoutRef.current);
      }
    };
  }, [isWebcamActive, models, processFrame]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-white mx-auto mb-4"></div>
          <p className="text-white text-lg">Loading AI models...</p>
          <p className="text-blue-200 text-sm mt-2">This may take a moment on first load</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 shadow-2xl">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main webcam view */}
          <div className="lg:col-span-2">
            <div className="relative bg-black rounded-xl overflow-hidden">
              <WebcamView
                videoRef={videoRef}
                canvasRef={canvasRef}
                isActive={isWebcamActive}
              />
              <DetectionOverlay detections={detections} />
              
              {/* Processing indicator */}
              {isProcessing && isWebcamActive && (
                <div className="absolute top-4 right-4 bg-blue-600/80 text-white px-3 py-1 rounded-full text-sm">
                  Processing...
                </div>
              )}
              
              {error && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/80">
                  <div className="text-center text-white p-4">
                    <p className="text-red-400 mb-4">{error}</p>
                    <button
                      onClick={startWebcam}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                    >
                      Try Again
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Control panel and stats */}
          <div className="space-y-6">
            <ControlPanel
              isWebcamActive={isWebcamActive}
              onStart={startWebcam}
              onStop={stopWebcam}
            />
            
            <StatsPanel
              detections={detections}
              fps={fps}
              isActive={isWebcamActive}
            />
            
            {/* Debug info */}
            {isWebcamActive && (
              <div className="bg-white/5 backdrop-blur-sm rounded-xl p-4">
                <h4 className="text-white text-sm font-medium mb-2">Debug Info</h4>
                <div className="text-xs text-blue-200 space-y-1">
                  <p>Processing: {isProcessing ? 'Yes' : 'No'}</p>
                  <p>Models loaded: {models ? 'Yes' : 'No'}</p>
                  <p>Canvas size: {canvasRef.current?.width || 0}x{canvasRef.current?.height || 0}</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmotionDetector;