import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './chat.html',
  styleUrls: ['./chat.scss'],
})
export class ChatComponent {
  question = '';
  answer = '';
  citations: Array<{
    source_path?: string;
    snippet?: string;
    page_number?: number;
    section_heading?: string;
  }> = [];
  loading = false;
  errorMessage = '';

  constructor(private http: HttpClient, private cdr: ChangeDetectorRef) {}

  askQuestion() {
    const q = this.question?.trim();
    if (!q) return;

    this.loading = true;
    this.answer = '';
    this.citations = [];
    this.errorMessage = '';

    this.http.post<any>('http://127.0.0.1:8000/chat', { question: q, top_k: 8 }).subscribe({
      next: (resp) => {
        this.answer = resp?.answer ?? '';
        this.citations = Array.isArray(resp?.citations) ? resp.citations : [];
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error('Error contacting backend:', err);
        this.errorMessage = 'Error contacting backend. Please try again.';
        this.answer = '';
        this.citations = [];
        this.loading = false;
        this.cdr.detectChanges();
      },
    });
  }

  openSource(citation: any) {
    const url = citation?.source_path;
    if (!url) {
      console.warn('No source_path found for citation:', citation);
      return;
    }
    window.open(url, '_blank', 'noopener,noreferrer');
  }

  citationTitle(c: any) {
    const parts: string[] = [];
    if (c?.source_path) parts.push(c.source_path);
    if (c?.section_heading) parts.push(c.section_heading);
    if (c?.page_number) parts.push(`Page ${c.page_number}`);
    return parts.join(' • ');
  }
}
