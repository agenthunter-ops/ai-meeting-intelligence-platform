import { Component } from '@angular/core';  // Angular core

@Component({
  selector: 'app-upload',
  templateUrl: './upload.component.html',  // HTML for upload UI
})
export class UploadComponent {
  taskId: string | null = null;            // store backend task id
  progress = 0;                             // progress percent

  onFileSelected(event: any) {
    const file = event.target.files[0];    // pick file from input
    const form = new FormData();
    form.append('file', file);             // append to form
    fetch('/api/upload', {                 // POST to backend
      method: 'POST',
      body: form
    })
    .then(res => res.json())
    .then(data => this.taskId = data.task_id);  // store id
  }

  pollStatus() {
    if (!this.taskId) return;
    fetch(`/api/status/${this.taskId}`)     // GET status
      .then(res => res.json())
      .then(s => this.progress = s.percent) // update progress
      .catch(() => {});                     // ignore errors
  }
}
