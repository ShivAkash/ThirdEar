"use client"

import React, { useState, useEffect, useRef } from 'react'
import { Mic, MicOff, Settings, Activity, ShieldAlert, Waves, Volume2, Info } from 'lucide-react'

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

// Type definitions
type TagResult = {
  label: string;
  score: number;
}

export default function AudioDashboard() {
  const [isRecording, setIsRecording] = useState(false)
  const [ancEnabled, setAncEnabled] = useState(false)
  const [mode, setMode] = useState<"audio_tagging" | "wake_word">("audio_tagging")
  const [wakewordThreshold, setWakewordThreshold] = useState(0.5)
  const [wakeWordDetected, setWakeWordDetected] = useState(false)
  const [wakeWordScore, setWakeWordScore] = useState(0)
  const [connectionError, setConnectionError] = useState<string | null>(null)

  // Audio Tagging State
  const [topTags, setTopTags] = useState<TagResult[]>([])

  // Refs
  const wsRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)

  // Canvas Ref for visualizer
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const animationFrameRef = useRef<number | undefined>(undefined)

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording()
    }
  }, [])

  // Switch mode logic
  useEffect(() => {
    if (isRecording) {
      stopRecording()
      startRecording(mode)
    }
  }, [mode, ancEnabled, wakewordThreshold])

  const startRecording = async (currentMode: string) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: currentMode === "audio_tagging" ? 32000 : 16000,
          echoCancellation: false,
          autoGainControl: false,
          noiseSuppression: false,
        }
      })

      mediaStreamRef.current = stream

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: currentMode === "audio_tagging" ? 32000 : 16000,
      })
      audioContextRef.current = audioCtx

      const source = audioCtx.createMediaStreamSource(stream)

      // Visualizer logic (Analyzer)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      analyserRef.current = analyser
      drawVisualizer()

      // ScriptProcessor for extracting raw PCM to send via WebSocket
      // Note: ScriptProcessor is deprecated but AudioWorklet requires more setup with separate files.
      // Using ScriptProcessor for simplicity and compatibility in a single file setup.
      const bufferSize = 4096
      const processor = audioCtx.createScriptProcessor(bufferSize, 1, 1)
      source.connect(processor)
      processor.connect(audioCtx.destination)
      processorRef.current = processor

      let wsUrl = "ws://localhost:8000/ws/" + currentMode + "?anc=" + ancEnabled.toString()
      if (currentMode === "wake_word") {
        wsUrl += "&threshold=" + wakewordThreshold.toString()
      }

      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        setIsRecording(true)
        setConnectionError(null)
        console.log("WebSocket connected to", currentMode)
      }

      ws.onerror = () => {
        setConnectionError(
          "Cannot connect to the Python backend at ws://localhost:8000. " +
          "Make sure you start it with: cd audioset_tagging_cnn && python -m uvicorn backend_api:app --reload --port 8000"
        )
        setIsRecording(false)
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === "tagging_results") {
            setTopTags(data.data)
          } else if (data.type === "wake_word_score") {
            setWakeWordScore(data.data.score)
          } else if (data.type === "wake_word_detected") {
            setWakeWordScore(data.data.score)
            triggerWakeWordAlert()
          } else if (data.type === "error") {
            console.error("Backend error:", data.message)
            setConnectionError("Backend error: " + data.message)
          }
        } catch (e) {
          console.error("Error parsing WS message", e)
        }
      }

      ws.onclose = () => {
        console.log("WebSocket closed")
        stopRecording()
      }

      // Send audio data
      processor.onaudioprocess = (e) => {
        if (ws.readyState === WebSocket.OPEN) {
          const inputData = e.inputBuffer.getChannelData(0)
          // We must send a copy or raw buffer
          // Float32Array directly sent as binary
          ws.send(inputData.buffer)
        }
      }

    } catch (err) {
      console.error("Error accessing microphone:", err)
      setIsRecording(false)
    }
  }

  const stopRecording = () => {
    setIsRecording(false)

    if (processorRef.current) {
      processorRef.current.disconnect()
      processorRef.current = null
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop())
      mediaStreamRef.current = null
    }

    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
    }
    clearCanvas()
  }

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording()
    } else {
      startRecording(mode)
    }
  }

  // Visualizer drawing logic
  const drawVisualizer = () => {
    const canvas = canvasRef.current
    const analyser = analyserRef.current
    if (!canvas || !analyser) return

    const canvasCtx = canvas.getContext('2d')
    if (!canvasCtx) return

    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)

    const draw = () => {
      animationFrameRef.current = requestAnimationFrame(draw)

      analyser.getByteFrequencyData(dataArray)

      canvasCtx.fillStyle = 'rgba(10, 10, 15, 0.2)'
      canvasCtx.fillRect(0, 0, canvas.width, canvas.height)

      const barWidth = (canvas.width / bufferLength) * 2.5
      let barHeight
      let x = 0

      for (let i = 0; i < bufferLength; i++) {
        barHeight = dataArray[i] / 2

        // Gradient coloring
        const gradient = canvasCtx.createLinearGradient(0, canvas.height - barHeight, 0, canvas.height)
        if (mode === 'audio_tagging') {
          gradient.addColorStop(0, '#3b82f6') // Blue
          gradient.addColorStop(1, '#8b5cf6') // Purple
        } else {
          gradient.addColorStop(0, wakeWordDetected ? '#ef4444' : '#10b981')
          gradient.addColorStop(1, wakeWordDetected ? '#b91c1c' : '#059669')
        }

        canvasCtx.fillStyle = gradient
        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight)

        x += barWidth + 1
      }
    }

    draw()
  }

  const clearCanvas = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const canvasCtx = canvas.getContext('2d')
    if (canvasCtx) {
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height)
      canvasCtx.fillStyle = 'rgba(10, 10, 15, 1)'
      canvasCtx.fillRect(0, 0, canvas.width, canvas.height)

      // Draw a flat line
      canvasCtx.fillStyle = '#4b5563'
      canvasCtx.fillRect(0, canvas.height - 2, canvas.width, 2)
    }
  }

  const triggerWakeWordAlert = () => {
    setWakeWordDetected(true)
    setTimeout(() => setWakeWordDetected(false), 2000)
  }

  const dynamicRecordingSpanClass = "w-2 h-2 rounded-full " + (isRecording ? "bg-emerald-500 animate-pulse" : "bg-rose-500");
  const dynamicButtonClass = "rounded-xl font-medium tracking-wide shadow-lg transition-all duration-300 " +
    (isRecording
      ? "bg-rose-500/10 text-rose-500 hover:bg-rose-500/20 hover:text-rose-400 border border-rose-500/20 hover:shadow-[0_0_20px_rgba(244,63,94,0.3)]"
      : "bg-indigo-600 hover:bg-indigo-500 text-white shadow-[0_0_20px_rgba(79,70,229,0.4)]");

  const dynamicAmbientBackgroundClass = "absolute inset-0 bg-gradient-to-t " +
    (mode === 'audio_tagging' ? 'from-indigo-500/5' : wakeWordDetected ? 'from-rose-500/20' : 'from-emerald-500/5') +
    " to-transparent transition-colors duration-1000";

  const dynamicWakeWordCardClass = "border-2 backdrop-blur-xl shadow-2xl transition-all duration-700 " +
    (wakeWordDetected ? "bg-rose-950/40 border-rose-500/50 shadow-[0_0_50px_rgba(244,63,94,0.3)] scale-[1.02]" : "bg-slate-900/40 border-slate-800/60");

  const dynamicWakeWordBlurClass = "absolute inset-0 rounded-full blur-xl transition-all duration-500 " +
    (wakeWordDetected ? 'bg-rose-500/40 opacity-100 scale-150' : 'bg-emerald-500/10 opacity-50 scale-100');

  const dynamicWakeWordInnerClass = "w-24 h-24 rounded-full border border-white/10 flex items-center justify-center relative z-10 transition-all duration-500 " +
    (wakeWordDetected ? 'bg-gradient-to-br from-rose-500 to-red-600 animate-pulse' : 'bg-slate-800/50');

  const dynamicWakeWordTextClass = "text-2xl font-black tracking-tight mb-2 transition-colors duration-300 " +
    (wakeWordDetected ? "text-rose-400" : "text-slate-300");

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-slate-200 p-4 md:p-8 font-sans transition-colors duration-500 selection:bg-indigo-500/30">

      <div className="max-w-5xl mx-auto space-y-8">
        {/* Header */}
        <header className="flex flex-col md:flex-row items-center justify-between gap-4 pb-6 border-b border-indigo-500/10">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-indigo-500/10 rounded-2xl border border-indigo-500/20 shadow-[0_0_20px_rgba(99,102,241,0.15)] relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-tr from-indigo-500/20 to-purple-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              <Activity className="w-8 h-8 text-indigo-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">Audio Detect</h1>
              <p className="text-slate-500 text-sm">Real-time Environmental Pattern Detection</p>
            </div>
          </div>

          {/* Master Control */}
          <div className="flex items-center gap-4 bg-slate-900/50 p-2 rounded-2xl border border-white/5 backdrop-blur-md">
            <div className="flex flex-col px-3">
              <span className="text-xs text-slate-500 font-medium tracking-wide uppercase">System Status</span>
              <span className="text-sm font-semibold flex items-center gap-2">
                <span className={dynamicRecordingSpanClass}></span>
                {isRecording ? 'Live Monitoring' : 'Offline'}
              </span>
            </div>
            <Button
              size="lg"
              onClick={toggleRecording}
              className={dynamicButtonClass}
            >
              {isRecording ? <><MicOff className="w-4 h-4 mr-2" /> Stop System</> : <><Mic className="w-4 h-4 mr-2" /> Initialize</>}
            </Button>
          </div>
        </header>

        <main className="grid grid-cols-1 lg:grid-cols-4 gap-6">

          {/* Main Visualizer and Data Area */}
          <div className="lg:col-span-3 space-y-6">

            {/* Audio Visualizer Card */}
            <Card className="bg-slate-900/40 border-slate-800/60 backdrop-blur-xl shadow-2xl overflow-hidden group">
              <CardContent className="p-0 relative">
                {/* Visualizer Canvas */}
                <div className="w-full h-48 bg-[#0a0a0f] relative overflow-hidden flex items-end">

                  {/* Background ambient light */}
                  <div className={dynamicAmbientBackgroundClass} />

                  <canvas
                    ref={canvasRef}
                    width={800}
                    height={192}
                    className="w-full h-full opacity-80 mix-blend-screen"
                  />

                  {/* Overlay Stats */}
                  <div className="absolute top-4 left-4 right-4 flex justify-between items-start pointer-events-none">
                    <div className="bg-black/50 backdrop-blur-md border border-white/5 rounded-full px-3 py-1 flex items-center gap-2 text-xs font-semibold tracking-wider text-slate-300">
                      <Waves className="w-3 h-3 text-indigo-400" />
                      FREQ DOMAIN
                    </div>
                    <div className="bg-black/50 backdrop-blur-md border border-white/5 rounded-full px-3 py-1 flex items-center gap-2 text-xs font-semibold tracking-wider text-slate-300">
                      <Volume2 className="w-3 h-3 text-indigo-400" />
                      {isRecording ? "ANALYZING" : "IDLE"}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Connection Error Banner */}
            {connectionError && (
              <div className="bg-rose-950/60 border border-rose-500/40 rounded-xl p-4 text-rose-300 text-sm flex items-start gap-3 animate-in fade-in slide-in-from-top-2 duration-300">
                <ShieldAlert className="w-5 h-5 shrink-0 mt-0.5 text-rose-400" />
                <div>
                  <p className="font-semibold text-rose-200 mb-1">Backend Not Connected</p>
                  <p className="text-rose-300/80">{connectionError}</p>
                </div>
                <button onClick={() => setConnectionError(null)} className="ml-auto text-rose-400 hover:text-rose-300 text-lg font-bold">&times;</button>
              </div>
            )}

            {/* Application Modes */}
            <Tabs defaultValue="audio_tagging" className="w-full flex flex-col" onValueChange={(val) => setMode(val as "audio_tagging" | "wake_word")}>
              <TabsList className="w-full bg-slate-900/60 border border-slate-800 p-1 rounded-2xl grid grid-cols-2 shadow-inner h-11">
                <TabsTrigger value="audio_tagging" className="rounded-xl data-active:bg-indigo-600 data-active:text-white transition-all duration-300 text-sm py-2">
                  Audio Tagging CNN
                </TabsTrigger>
                <TabsTrigger value="wake_word" className="rounded-xl data-active:bg-emerald-600 data-active:text-white transition-all duration-300 text-sm py-2">
                  OpenWakeWord
                </TabsTrigger>
              </TabsList>

              <div className="mt-6">
                <TabsContent value="audio_tagging" className="space-y-4 outline-none animate-in fade-in slide-in-from-bottom-4 duration-500">
                  <Card className="bg-slate-900/40 border-slate-800/60 backdrop-blur-xl shrink-0 h-[420px] overflow-hidden flex flex-col">
                    <CardHeader className="border-b border-slate-800/60 pb-4 shrink-0">
                      <div className="flex items-center justify-between">
                        <div>
                          <CardTitle className="text-xl font-bold flex items-center gap-2">
                            <Activity className="w-5 h-5 text-indigo-400" />
                            Detected Signatures
                          </CardTitle>
                          <CardDescription className="text-slate-400 mt-1">Real-time classification probability</CardDescription>
                        </div>
                        <div className="text-right">
                          <span className="text-2xl font-black text-indigo-400 tracking-tighter">Top 10</span>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="pt-6 overflow-y-auto custom-scrollbar flex-1 relative">
                      {!isRecording && topTags.length === 0 && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 p-6 text-center animate-pulse">
                          <MicOff className="w-12 h-12 mb-4 text-slate-700" />
                          <p>System offline. Initialize monitoring to view real-time audio signatures.</p>
                        </div>
                      )}
                      <div className="space-y-5">
                        {topTags.map((tag, i) => (
                          <div key={i} className="space-y-2 group">
                            <div className="flex justify-between items-end text-sm">
                              <span className="font-semibold text-slate-200 group-hover:text-indigo-300 transition-colors drop-shadow-md">{tag.label}</span>
                              <span className="font-mono text-indigo-400/80">{(tag.score * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden relative">
                              <div
                                className="absolute top-0 left-0 h-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-300 ease-out"
                                style={{ width: Math.max(2, tag.score * 100).toString() + "%" }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="wake_word" className="outline-none animate-in fade-in slide-in-from-bottom-4 duration-500">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Wake word status */}
                    <Card className={dynamicWakeWordCardClass}>
                      <CardContent className="p-8 flex flex-col items-center justify-center min-h-[300px] text-center space-y-6">
                        <div className="relative">
                          <div className={dynamicWakeWordBlurClass} />
                          <div className={dynamicWakeWordInnerClass}>
                            {wakeWordDetected ? <ShieldAlert className="w-10 h-10 text-white" /> : <ShieldAlert className="w-10 h-10 text-emerald-500/50" />}
                          </div>
                        </div>

                        <div>
                          <h3 className={dynamicWakeWordTextClass}>
                            {wakeWordDetected ? "Target Acquired" : "Scanning..."}
                          </h3>
                          <p className="text-slate-400 font-medium">Listening for <span className="text-white">&quot;Shiv Akash&quot;</span></p>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Wake word settings */}
                    <Card className="bg-slate-900/40 border-slate-800/60 backdrop-blur-xl">
                      <CardHeader>
                        <CardTitle className="text-lg flex items-center gap-2">
                          <Settings className="w-4 h-4 text-emerald-400" />
                          Detection Telemetry
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-8">
                        <div className="space-y-4">
                          <div className="flex justify-between items-center text-sm font-medium">
                            <span className="text-slate-400">Current Confidence</span>
                            <span className="font-mono text-emerald-400">{(wakeWordScore * 100).toFixed(1)}%</span>
                          </div>
                          <Progress value={Math.max(5, wakeWordScore * 100)} className="h-3 bg-slate-800" indicatorclassname="bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-200" />
                        </div>

                        <div className="space-y-6">
                          <div className="flex justify-between items-center">
                            <label className="text-sm font-semibold text-slate-300">Sensitivity Threshold</label>
                            <span className="px-2 py-1 bg-slate-800 rounded text-xs font-mono text-slate-300">{wakewordThreshold.toFixed(2)}</span>
                          </div>
                          <Slider
                            defaultValue={[0.5]}
                            max={1}
                            min={0.1}
                            step={0.05}
                            onValueChange={(vals: number | readonly number[]) => {
                              if (Array.isArray(vals) && vals.length > 0) {
                                setWakewordThreshold(vals[0])
                              } else if (typeof vals === 'number') {
                                setWakewordThreshold(vals)
                              }
                            }}
                            className="cursor-pointer"
                          />
                          <p className="text-xs text-slate-500 flex items-start gap-2">
                            <Info className="w-4 h-4 shrink-0 mt-0.5" />
                            Lower values increase detection rate but may cause more false positives.
                          </p>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>
              </div>
            </Tabs>
          </div>

          {/* Sidebar / Configuration */}
          <div className="lg:col-span-1 space-y-6">
            <Card className="bg-slate-900/40 border-slate-800/60 backdrop-blur-xl shadow-xl h-full sticky top-8">
              <CardHeader className="pb-4 border-b border-slate-800/60">
                <CardTitle className="text-lg font-bold flex items-center gap-2">
                  <Settings className="w-5 h-5 text-indigo-400" />
                  Engine Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-6 space-y-8">

                {/* ANC Setting */}
                <div className="space-y-4 bg-slate-900/50 p-4 rounded-xl border border-white/5 shadow-inner">
                  <div className="flex items-center justify-between">
                    <label htmlFor="anc-toggle" className="text-sm font-semibold text-slate-200">Active Noise Cancel</label>
                    <Switch
                      id="anc-toggle"
                      checked={ancEnabled}
                      onCheckedChange={setAncEnabled}
                      className="data-[state=checked]:bg-indigo-500"
                    />
                  </div>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    Applies <code className="text-indigo-400 bg-indigo-500/10 px-1 rounded">noisereduce</code> to incoming audio before feeding into the tensor models. Useful for noisy environments.
                  </p>
                </div>

                {/* Info Block */}
                <div className="pt-4 border-t border-slate-800">
                  <h4 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-3">System Specs</h4>
                  <ul className="space-y-3 text-sm text-slate-400">
                    <li className="flex justify-between">
                      <span>Input Source</span>
                      <span className="font-mono text-slate-300">Web Audio API</span>
                    </li>
                    <li className="flex justify-between">
                      <span>Sample Rate</span>
                      <span className="font-mono text-slate-300">{mode === 'audio_tagging' ? '32 kHz' : '16 kHz'}</span>
                    </li>
                    <li className="flex justify-between">
                      <span>Model Sync</span>
                      <span className="font-mono text-slate-300">WebSocket</span>
                    </li>
                  </ul>
                </div>

              </CardContent>
            </Card>
          </div>

        </main>
      </div>

      {/* Global custom CSS for scrollbar override */}
      <style dangerouslySetInnerHTML={{
        __html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(15, 23, 42, 0.5);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(99, 102, 241, 0.3);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(99, 102, 241, 0.6);
        }
      `}} />
    </div>
  )
}
