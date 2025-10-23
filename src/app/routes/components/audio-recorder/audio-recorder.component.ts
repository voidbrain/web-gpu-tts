import { Component, ElementRef, Output, ViewChild, EventEmitter } from '@angular/core';
import { NotificationService } from '@service/notification.service';
import { webmFixDuration } from 'app/utils/blob-fix';
import { TranslatePipe, TranslateService } from '@ngx-translate/core';

@Component({
  selector: 'app-audio-recorder',
  templateUrl: './audio-recorder.component.html',
  standalone: true,
  imports: [TranslatePipe],
})
export class AudioRecorderComponent {
  isRecording = false;
  private mediaRecorder?: MediaRecorder;
  private canvasCtx!: CanvasRenderingContext2D;
  private analyser?: AnalyserNode;
  private silenceStartTime: number | null = null;
  private silenceInterval?: number;

  private readonly SILENCE_THRESHOLD = 0.01; // amplitude threshold
  private readonly SILENCE_TIMEOUT = 2000; // 2 seconds

  @ViewChild('soundClips') soundClips!: ElementRef<HTMLDivElement>;
  @ViewChild('visualizer') visualizer!: ElementRef<HTMLCanvasElement>;

  @Output() recordingStarted = new EventEmitter<boolean>();
  @Output() recordingComplete = new EventEmitter<{ audioBuffer: Blob; audioUrl: string }>();

  constructor(
    private translateService: TranslateService,
    private notificationService: NotificationService
  ) {}

  private getMimeType(): string | undefined {
    const types = ['audio/webm', 'audio/mp4', 'audio/ogg', 'audio/wav', 'audio/aac'];
    return types.find(type => MediaRecorder.isTypeSupported(type));
  }

  async startRecording(): Promise<void> {
    this.isRecording = true;
    this.recordingStarted.emit(true);
    const audioChunks: Blob[] = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.visualize(stream);

      const startTime = Date.now();
      const mimeType = this.getMimeType();
      if (!mimeType) throw new Error('No supported mime type for MediaRecorder');

      this.mediaRecorder = new MediaRecorder(stream, { mimeType });

      this.mediaRecorder.ondataavailable = async event => {
        if (event.data.size === 0) return;
        audioChunks.push(event.data);

        // fix webm duration if needed
        if (this.mediaRecorder?.state === 'inactive' && mimeType === 'audio/webm') {
          const duration = Date.now() - startTime;
          const fixedBlob = await webmFixDuration(new Blob(audioChunks, { type: mimeType }), duration, mimeType);
          this.completeRecording(fixedBlob);
        }
      };

      this.mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunks, { type: mimeType });
        this.completeRecording(blob);
      };

      this.mediaRecorder.start();
      this.startSilenceDetection();
    } catch (error) {
      this.isRecording = false;
      this.recordingStarted.emit(false);
      this.notificationService.showError(this.translateService.instant('ERR_ACCESSING_MICROPHONE'));
      console.error(error);
    }
  }

  stopRecording(): void {
    if (!this.isRecording) return;
    this.isRecording = false;
    this.mediaRecorder?.stop();
    this.stopSilenceDetection();
  }

  private completeRecording(audioBlob: Blob): void {
    const audioUrl = URL.createObjectURL(audioBlob);
    this.recordingComplete.emit({ audioBuffer: audioBlob, audioUrl });
  }

  private visualize(stream: MediaStream): void {
    this.setupCanvas();

    const audioCtx = new AudioContext();
    const source = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);
    this.analyser = analyser;

    this.draw(analyser);
  }

  private setupCanvas() {
    if (!this.visualizer) return;
    this.canvasCtx = this.visualizer.nativeElement.getContext('2d')!;
    window.onresize = () => (this.visualizer.nativeElement.width = this.soundClips.nativeElement.offsetWidth);
    window.dispatchEvent(new Event('resize'));
  }

  private draw(analyser: AnalyserNode) {
    const canvas = this.visualizer.nativeElement;
    const WIDTH = canvas.width;
    const HEIGHT = canvas.height;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const drawVisualizer = () => {
      requestAnimationFrame(drawVisualizer);
      analyser.getByteTimeDomainData(dataArray);

      // draw waveform
      this.canvasCtx.fillStyle = 'rgb(200,200,200)';
      this.canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);
      this.canvasCtx.lineWidth = 2;
      this.canvasCtx.strokeStyle = 'rgb(0,0,0)';
      this.canvasCtx.beginPath();

      let x = 0;
      const sliceWidth = WIDTH / bufferLength;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * HEIGHT) / 2;
        if (i === 0) this.canvasCtx.moveTo(x, y);
        else this.canvasCtx.lineTo(x, y);
        x += sliceWidth;
      }
      this.canvasCtx.lineTo(WIDTH, HEIGHT / 2);
      this.canvasCtx.stroke();
    };

    drawVisualizer();
  }

  // --- silence detection logic ---
  private startSilenceDetection() {
    this.silenceStartTime = null;
    this.silenceInterval = window.setInterval(() => this.checkSilence(), 100);
  }

  private stopSilenceDetection() {
    if (this.silenceInterval) {
      clearInterval(this.silenceInterval);
      this.silenceInterval = undefined;
    }
  }

  private checkSilence() {
    if (!this.analyser || !this.isRecording) return;

    const bufferLength = this.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    this.analyser.getByteTimeDomainData(dataArray);

    let maxAmplitude = 0;
    for (let i = 0; i < bufferLength; i++) {
      const amp = Math.abs(dataArray[i] / 128.0 - 1);
      if (amp > maxAmplitude) maxAmplitude = amp;
    }

    const now = Date.now();
    if (maxAmplitude < this.SILENCE_THRESHOLD) {
      if (!this.silenceStartTime) this.silenceStartTime = now;
      else if (now - this.silenceStartTime >= this.SILENCE_TIMEOUT) {
        this.stopRecording();
        this.silenceStartTime = null;
      }
    } else {
      this.silenceStartTime = null;
    }
  }
}
