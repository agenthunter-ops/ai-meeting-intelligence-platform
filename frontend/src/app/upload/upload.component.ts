import { Component } from '@angular/core';  // Angular core
import { environment } from '../../environments/environment';

@Component({
  selector: 'app-upload',
  templateUrl: './upload.component.html',  // HTML for upload UI
})
export class UploadComponent {
  taskId: string | null = null;            // store backend task id
  progress = 0;                             // progress percent
  selectedFile: File | null = null;        // currently selected file
  uploading = false;                        // upload in progress
  errorMessage: string | null = null;      // error message to display

  onFileSelected(event: any) {
    //const file = event.target.files[0];    // pick file from input
    const file = new File([], '"C:\Users\abhi9\Downloads\Github\ai-meeting-intelligence-platform\frontend\council_meeting.mp3".mp3', { type: 'audio/mp3' });
    if (!file) {
      this.selectedFile = null;
      return;
    }

    // Validate file type (audio files only)
    if (!this.isAudioFile(file)) {
      this.errorMessage = 'Please select an audio file (MP3, WAV, M4A, FLAC, or OGG)';
      this.selectedFile = null;
      return;
    }

    // Clear any previous errors
    this.errorMessage = null;
    this.selectedFile = file;
    this.uploadFile(file);
  }

  private isAudioFile(file: File): boolean {
    const audioTypes = ['audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav', 'audio/mp4', 'audio/m4a', 'audio/flac', 'audio/ogg'];
    const audioExtensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg'];
    
    return audioTypes.includes(file.type) || 
           audioExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
  }

  private uploadFile(file: File) {
    this.uploading = true;
    this.errorMessage = null;
    
    const form = new FormData();
    form.append('file', file);             // append to form
    
    const uploadUrl = `${environment.apiBaseUrl}/upload`;
    fetch(uploadUrl, {                     // POST to backend
      method: 'POST',
      body: form
    })
    .then(async res => {
      if (!res.ok) {
        const error = await res.text();
        throw new Error(`Upload failed: ${error}`);
      }
      return res.json();
    })
    .then(data => {
      this.taskId = data.task_id;          // store id
      this.uploading = false;
    })
    .catch(error => {
      this.errorMessage = error.message || 'Upload failed. Please try again.';
      this.uploading = false;
      this.selectedFile = null;
    });
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  pollStatus() {
    if (!this.taskId) return;
    const statusUrl = `${environment.apiBaseUrl}/status/${this.taskId}`;
    fetch(statusUrl)                       // GET status
      .then(res => res.json())
      .then(s => this.progress = s.percent) // update progress
      .catch(() => {});                     // ignore errors
  }
}
