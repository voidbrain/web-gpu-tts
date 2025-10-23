import { Component } from '@angular/core';
import { BsModalService, ModalModule } from 'ngx-bootstrap/modal';
import { TranslatePipe } from '@ngx-translate/core';
import { DeviceDetectorService } from 'ngx-device-detector';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { TranscriberSettingsModalComponent } from '@routes/components/transcriber-settings-modal/transcriber-settings-modal.component';
import { LanguageComponent } from '@shared/components/language/language.component';
import { AudioRecorderComponent } from '@routes/components/audio-recorder/audio-recorder.component';
import { AudioPlayerComponent } from '@routes/components/audio-player/audio-player.component';
import { TranscriberComponent } from '@routes/components/transcriber/transcriber.component';
import { ModelCardComponent } from '@routes/components/model-card/model-card.component';
import { LoggerService } from '@service/logger.service';
import { EmbedderService } from '@service/command-parser/command-parser.service';
// import { CommandService } from '@service/command-parser/command-parser.service';

@Component({
  selector: 'app-audio-manager',
  templateUrl: './audio-manager.component.html',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ModalModule,
    LanguageComponent,
    AudioRecorderComponent,
    AudioPlayerComponent,
    TranscriberComponent,
    ModelCardComponent,
    TranslatePipe,
  ],
  providers: [BsModalService],
})
export class AudioManagerComponent {
  IS_WEBGPU_AVAILABLE = !!(navigator as any).gpu;
  isTranscribeEnabled = false;
  isTranscriptionInProgress = false;
  audioUrl?: string;
  audioBlob?: Blob;
  progress = 0;

  inputText = 'Charge battery 12 yellow series';
  command: any | null = null;

  constructor(
    private modalService: BsModalService,
    protected deviceDetectorService: DeviceDetectorService,
    private embedderService: EmbedderService
  ) {
    // this.commandService.getInstance().then(() => {
    //   // this.commandService.parseCommand(transcribedText).then(result => {
    //   //   this.commandService.executeCommand(result as any);
    //   // });
    // });
    this.init();
  }

  async onTranscriptionComplete(transcribedText: string) {
    this.inputText = transcribedText; // optional: show in input
    console.log('Transcribed text:', transcribedText);

    // Pass to EmbedderService / command parser
    this.command = await this.embedderService.parseCommand(transcribedText);
    console.log('Parsed command:', this.command);

    // Optionally, execute command
    // await this.embedderService.executeCommand(this.command);
  }

  async init() {
    // await this.commandService.init();
    // const transcribedText = 'Sample transcribed text'; // Replace with actual transcribed text
    // const result = await this.commandService.parseCommand(transcribedText);
    // this.commandService.executeCommand(result as any);
    await this.embedderService.init();
  }

  async embedText() {
    this.command = await this.embedderService.parseCommand(this.inputText);
    console.log('command:', this.command);
  }

  onRecordingStarted(): void {
    LoggerService.logInfo('hello');
    this.reset();
  }

  onRecordingComplete(data: { audioBuffer: Blob; audioUrl: string }): void {
    this.audioUrl = data.audioUrl;
    this.audioBlob = data.audioBuffer;
    this.transcribe();
  }

  clearCurrentRecording(): void {
    this.audioUrl = '';
    this.audioBlob = undefined;
  }

  openSettings(): void {
    this.modalService.show(TranscriberSettingsModalComponent, {
      class: 'modal-lg modal-dialog-centered',
      ignoreBackdropClick: true,
    });
  }

  transcribe(): void {
    this.isTranscribeEnabled = true;
  }

  reset(): void {
    this.isTranscribeEnabled = false;
    this.clearCurrentRecording();
  }

  setTranscriptionInProgress(isTranscriptionInProgress: boolean) {
    this.isTranscriptionInProgress = isTranscriptionInProgress;
  }
}
