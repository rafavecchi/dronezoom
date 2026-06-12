// Google Drive upload via Google Identity Services + Drive v3 resumable
// upload. Uses the minimal drive.file scope: the app can only see files
// it created. Requires a (free) OAuth client ID — see README.

const SCOPE = 'https://www.googleapis.com/auth/drive.file';
const LS_KEY = 'dronezoom.gdriveClientId';

interface TokenResponse {
  access_token: string;
  expires_in?: number;
  error?: string;
}

interface TokenClient {
  requestAccessToken(opts?: { prompt?: string }): void;
}

interface GisOauth2 {
  initTokenClient(cfg: {
    client_id: string;
    scope: string;
    callback: (resp: TokenResponse) => void;
    error_callback?: (err: { type?: string; message?: string }) => void;
  }): TokenClient;
}

function gis(): GisOauth2 {
  const g = (
    window as unknown as { google?: { accounts?: { oauth2?: GisOauth2 } } }
  ).google?.accounts?.oauth2;
  if (!g) {
    throw new Error('Google sign-in script not loaded (check network / ad blockers)');
  }
  return g;
}

export function getClientId(): string | null {
  return localStorage.getItem(LS_KEY);
}

export function setClientId(id: string): void {
  localStorage.setItem(LS_KEY, id.trim());
}

let cached: { token: string; expiresAt: number } | null = null;

function getAccessToken(): Promise<string> {
  if (cached && Date.now() < cached.expiresAt - 60_000) {
    return Promise.resolve(cached.token);
  }
  const clientId = getClientId();
  if (!clientId) return Promise.reject(new Error('No OAuth client ID configured'));
  return new Promise((resolve, reject) => {
    const tc = gis().initTokenClient({
      client_id: clientId,
      scope: SCOPE,
      callback: (resp) => {
        if (resp.error || !resp.access_token) {
          reject(new Error(resp.error ?? 'No access token returned'));
          return;
        }
        cached = {
          token: resp.access_token,
          expiresAt: Date.now() + (resp.expires_in ?? 3600) * 1000,
        };
        resolve(resp.access_token);
      },
      error_callback: (err) =>
        reject(new Error(err.message ?? err.type ?? 'Google sign-in was cancelled')),
    });
    tc.requestAccessToken();
  });
}

export interface DriveFile {
  id: string;
  link: string;
}

export async function uploadToDrive(
  blob: Blob,
  name: string,
  onProgress: (frac: number) => void,
): Promise<DriveFile> {
  const token = await getAccessToken();

  const init = await fetch(
    'https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable',
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json; charset=UTF-8',
        'X-Upload-Content-Type': 'video/mp4',
        'X-Upload-Content-Length': String(blob.size),
      },
      body: JSON.stringify({ name, mimeType: 'video/mp4' }),
    },
  );
  if (!init.ok) {
    throw new Error(`Drive upload init failed: ${init.status} ${await init.text()}`);
  }
  const sessionUrl = init.headers.get('Location');
  if (!sessionUrl) throw new Error('Drive did not return an upload session URL');

  // XHR for upload progress events
  const fileId = await new Promise<string>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', sessionUrl);
    xhr.setRequestHeader('Content-Type', 'video/mp4');
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve((JSON.parse(xhr.responseText) as { id: string }).id);
        } catch {
          reject(new Error('Unexpected Drive response'));
        }
      } else {
        reject(new Error(`Drive upload failed: ${xhr.status} ${xhr.responseText}`));
      }
    };
    xhr.onerror = () => reject(new Error('Network error during Drive upload'));
    xhr.send(blob);
  });

  return { id: fileId, link: `https://drive.google.com/file/d/${fileId}/view` };
}
