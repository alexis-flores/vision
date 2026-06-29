# assets

Drop your brand logo here as **`logo.png`** and it will appear automatically in
the **top-right** of the live-view window (`gui_bridge.CameraViewer`).

- Format: PNG (a **transparent background** looks best — it sits on the dark HUD
  panel).
- Size: any; it's scaled to **40 px tall** (aspect ratio preserved). A wide
  wordmark or a small square mark both work.
- No code change needed — if `assets/logo.png` exists it's shown; if it's
  absent, nothing appears.

To use a different file or location, pass `logo_path=...` to `CameraViewer(...)`
(or change `gui_bridge.LOGO_PATH`).
