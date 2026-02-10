/*
 * dictate.c — Sherpa-ONNX dictation in C.
 *
 * Single-file, zero Python dependencies. Links against libsherpa-onnx-c-api
 * and libportaudio. Uses Parakeet-TDT 0.6B v3 int8 + Silero VAD.
 *
 * Build:  make
 * Run:    ./dictate
 */

#include <ctype.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <X11/Xlib.h>
#include <X11/XKBlib.h>
#include <X11/keysym.h>
#include <X11/extensions/XTest.h>
#include <portaudio.h>
#include <sherpa-onnx/c-api/c-api.h>

/* ── Constants (matching Python version) ──────────────────────────────────── */

#define SAMPLE_RATE      16000
#define NUM_THREADS      8
#define VAD_THRESHOLD    0.5f
#define VAD_MIN_SILENCE  0.4f
#define VAD_MIN_SPEECH   0.3f
#define VAD_MAX_SPEECH   5.0f
#define VAD_WINDOW_SIZE  512
#define MAX_QUEUE_SIZE   5

/* Model paths (relative to binary location) */
#define MODEL_DIR  "models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
#define VAD_MODEL  "models/silero_vad.onnx"

/* ── Audio segment + ring buffer queue ────────────────────────────────────── */

typedef struct {
    float *samples;
    int32_t n;
} AudioSegment;

typedef struct {
    AudioSegment items[MAX_QUEUE_SIZE];
    int head;
    int tail;
    int count;
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
} SegmentQueue;

static void queue_init(SegmentQueue *q) {
    memset(q, 0, sizeof(*q));
    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->cond, NULL);
}

static void queue_destroy(SegmentQueue *q) {
    /* Free any remaining segments */
    for (int i = 0; i < q->count; i++) {
        int idx = (q->head + i) % MAX_QUEUE_SIZE;
        free(q->items[idx].samples);
    }
    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->cond);
}

static void queue_push(SegmentQueue *q, float *samples, int32_t n) {
    pthread_mutex_lock(&q->mutex);
    if (q->count == MAX_QUEUE_SIZE) {
        /* Drop oldest */
        free(q->items[q->head].samples);
        q->head = (q->head + 1) % MAX_QUEUE_SIZE;
        q->count--;
    }
    q->items[q->tail].samples = samples;
    q->items[q->tail].n = n;
    q->tail = (q->tail + 1) % MAX_QUEUE_SIZE;
    q->count++;
    pthread_cond_signal(&q->cond);
    pthread_mutex_unlock(&q->mutex);
}

/* Returns 1 if got a segment, 0 on timeout/shutdown */
static int queue_pop(SegmentQueue *q, AudioSegment *out, volatile int *running) {
    pthread_mutex_lock(&q->mutex);
    while (q->count == 0 && *running) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 500000000L; /* 500ms */
        if (ts.tv_nsec >= 1000000000L) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000L;
        }
        pthread_cond_timedwait(&q->cond, &q->mutex, &ts);
    }
    if (q->count == 0) {
        pthread_mutex_unlock(&q->mutex);
        return 0;
    }
    *out = q->items[q->head];
    q->head = (q->head + 1) % MAX_QUEUE_SIZE;
    q->count--;
    pthread_mutex_unlock(&q->mutex);
    return 1;
}

/* ── Globals ──────────────────────────────────────────────────────────────── */

static volatile int g_running = 1;
static const SherpaOnnxOfflineRecognizer *g_recognizer = NULL;
static const SherpaOnnxVoiceActivityDetector *g_vad = NULL;
static SegmentQueue g_queue;

/* Audio buffer for accumulating samples before feeding VAD */
static float  g_audio_buf[SAMPLE_RATE]; /* up to 1s buffer */
static int    g_audio_buf_len = 0;
static pthread_mutex_t g_vad_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Path prefix for models (set from argv[0] location) */
static char g_basedir[4096];

/* ── XTest direct text injection (no fork, no clipboard, single flush) ────── */

static Display *g_dpy = NULL;
static KeyCode  g_shift;

static void x11_init(void) {
    g_dpy = XOpenDisplay(NULL);
    if (!g_dpy) {
        fprintf(stderr, "ERROR: Cannot open X11 display ($DISPLAY=%s)\n",
                getenv("DISPLAY") ? getenv("DISPLAY") : "unset");
        exit(1);
    }
    int ev, er, maj, min;
    if (!XTestQueryExtension(g_dpy, &ev, &er, &maj, &min)) {
        fprintf(stderr, "ERROR: XTest extension not available.\n");
        exit(1);
    }
    g_shift = XKeysymToKeycode(g_dpy, XK_Shift_L);
}

static void x11_cleanup(void) {
    if (g_dpy) { XCloseDisplay(g_dpy); g_dpy = NULL; }
}

static void type_text(const char *text) {
    if (!g_dpy || !text || !text[0]) return;

    for (const char *p = text; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (c < 0x20 || c > 0x7e) continue;

        KeySym ks = (KeySym)c;
        KeyCode kc = XKeysymToKeycode(g_dpy, ks);
        if (kc == 0) continue;

        /* Check if shift needed: compare with unshifted keysym at this keycode */
        int need_shift = (XkbKeycodeToKeysym(g_dpy, kc, 0, 0) != ks);

        if (need_shift) XTestFakeKeyEvent(g_dpy, g_shift, True, 0);
        XTestFakeKeyEvent(g_dpy, kc, True, 0);
        XTestFakeKeyEvent(g_dpy, kc, False, 0);
        if (need_shift) XTestFakeKeyEvent(g_dpy, g_shift, False, 0);
    }
    XFlush(g_dpy);
}

/* ── Garbage filter ───────────────────────────────────────────────────────── */

static int is_garbage(const char *text) {
    if (!text) return 1;

    /* Skip leading whitespace */
    while (*text && isspace((unsigned char)*text)) text++;
    if (!*text) return 1;

    int len = (int)strlen(text);
    /* Trim trailing whitespace for length check */
    while (len > 0 && isspace((unsigned char)text[len - 1])) len--;
    if (len == 0) return 1;

    /* Check unique chars and alnum presence */
    int has_alnum = 0;
    char seen[256] = {0};
    int unique = 0;
    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (isalnum(c)) has_alnum = 1;
        if (!seen[c]) { seen[c] = 1; unique++; }
    }

    /* Reject: <=2 unique chars and no alphanumeric */
    if (unique <= 2 && !has_alnum) return 1;
    /* Reject: short (<3 chars) with no alphanumeric */
    if (len < 3 && !has_alnum) return 1;

    return 0;
}

/* ── PortAudio callback ───────────────────────────────────────────────────── */

static int pa_callback(const void *input, void *output,
                       unsigned long frame_count,
                       const PaStreamCallbackTimeInfo *time_info,
                       PaStreamCallbackFlags status_flags,
                       void *user_data)
{
    (void)output; (void)time_info; (void)status_flags; (void)user_data;
    const float *in = (const float *)input;
    if (!in) return paContinue;

    pthread_mutex_lock(&g_vad_mutex);

    /* Accumulate into buffer */
    int to_copy = (int)frame_count;
    if (g_audio_buf_len + to_copy > SAMPLE_RATE)
        to_copy = SAMPLE_RATE - g_audio_buf_len;
    memcpy(g_audio_buf + g_audio_buf_len, in, to_copy * sizeof(float));
    g_audio_buf_len += to_copy;

    /* Feed VAD in VAD_WINDOW_SIZE chunks */
    while (g_audio_buf_len >= VAD_WINDOW_SIZE) {
        SherpaOnnxVoiceActivityDetectorAcceptWaveform(g_vad, g_audio_buf, VAD_WINDOW_SIZE);

        /* Shift buffer */
        g_audio_buf_len -= VAD_WINDOW_SIZE;
        if (g_audio_buf_len > 0)
            memmove(g_audio_buf, g_audio_buf + VAD_WINDOW_SIZE, g_audio_buf_len * sizeof(float));
    }

    /* Extract completed speech segments */
    while (!SherpaOnnxVoiceActivityDetectorEmpty(g_vad)) {
        const SherpaOnnxSpeechSegment *seg = SherpaOnnxVoiceActivityDetectorFront(g_vad);
        float duration = (float)seg->n / SAMPLE_RATE;

        if (duration >= VAD_MIN_SPEECH) {
            /* Copy samples and enqueue */
            float *copy = (float *)malloc(seg->n * sizeof(float));
            if (copy) {
                memcpy(copy, seg->samples, seg->n * sizeof(float));
                /* queue_push takes ownership of copy */
                queue_push(&g_queue, copy, seg->n);
                fprintf(stderr, "\r\033[K  [detected %.1fs speech]", duration);
                fflush(stderr);
            }
        }

        SherpaOnnxDestroySpeechSegment(seg);
        SherpaOnnxVoiceActivityDetectorPop(g_vad);
    }

    pthread_mutex_unlock(&g_vad_mutex);
    return paContinue;
}

/* ── Transcription worker thread ──────────────────────────────────────────── */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void *transcription_worker(void *arg) {
    (void)arg;
    AudioSegment seg;

    while (g_running) {
        if (!queue_pop(&g_queue, &seg, &g_running))
            continue;

        float duration = (float)seg.n / SAMPLE_RATE;
        fprintf(stderr, "\r\033[K  [transcribing %.1fs...]", duration);
        fflush(stderr);

        double t0 = now_ms();
        const SherpaOnnxOfflineStream *stream = SherpaOnnxCreateOfflineStream(g_recognizer);
        SherpaOnnxAcceptWaveformOffline(stream, SAMPLE_RATE, seg.samples, seg.n);
        SherpaOnnxDecodeOfflineStream(g_recognizer, stream);
        double t1 = now_ms();

        const SherpaOnnxOfflineRecognizerResult *r = SherpaOnnxGetOfflineStreamResult(stream);
        if (r && r->text && !is_garbage(r->text)) {
            /* Trim leading whitespace */
            const char *text = r->text;
            while (*text && isspace((unsigned char)*text)) text++;

            /* Append space and type */
            size_t tlen = strlen(text);
            char *buf = (char *)malloc(tlen + 2);
            if (buf) {
                memcpy(buf, text, tlen);
                buf[tlen] = ' ';
                buf[tlen + 1] = '\0';
                double t2 = now_ms();
                type_text(buf);
                double t3 = now_ms();
                fprintf(stderr, "\r\033[K  >> %s\n", text);
                fprintf(stderr, "     [%.1fs audio | recognize: %.0fms | type: %.0fms | total: %.0fms]\n",
                        duration, t1 - t0, t3 - t2, t3 - t0);
                free(buf);
            }
        } else {
            fprintf(stderr, "\r\033[K");
            fflush(stderr);
        }

        if (r) SherpaOnnxDestroyOfflineRecognizerResult(r);
        SherpaOnnxDestroyOfflineStream(stream);
        free(seg.samples);
    }
    return NULL;
}

/* ── Signal handler ───────────────────────────────────────────────────────── */

static void on_signal(int sig) {
    (void)sig;
    g_running = 0;
}

/* ── Resolve base directory from argv[0] ──────────────────────────────────── */

static void resolve_basedir(const char *argv0) {
    /* Try /proc/self/exe first (Linux) */
    ssize_t len = readlink("/proc/self/exe", g_basedir, sizeof(g_basedir) - 1);
    if (len > 0) {
        g_basedir[len] = '\0';
    } else {
        /* Fallback: use argv[0] */
        if (argv0[0] == '/') {
            snprintf(g_basedir, sizeof(g_basedir), "%s", argv0);
        } else {
            char cwd[2048];
            if (getcwd(cwd, sizeof(cwd)))
                snprintf(g_basedir, sizeof(g_basedir), "%s/%s", cwd, argv0);
            else
                snprintf(g_basedir, sizeof(g_basedir), "%s", argv0);
        }
    }
    /* Strip filename to get directory */
    char *slash = strrchr(g_basedir, '/');
    if (slash) *slash = '\0';
}

static void make_path(char *dst, size_t dstsz, const char *rel) {
    snprintf(dst, dstsz, "%s/%s", g_basedir, rel);
}

/* ── Main ─────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    (void)argc;
    resolve_basedir(argv[0]);

    /* Build model paths */
    char encoder[8192], decoder[8192], joiner[8192], tokens[8192], vad_model[8192];
    make_path(encoder,   sizeof(encoder),   MODEL_DIR "/encoder.int8.onnx");
    make_path(decoder,   sizeof(decoder),   MODEL_DIR "/decoder.int8.onnx");
    make_path(joiner,    sizeof(joiner),    MODEL_DIR "/joiner.int8.onnx");
    make_path(tokens,    sizeof(tokens),    MODEL_DIR "/tokens.txt");
    make_path(vad_model, sizeof(vad_model), VAD_MODEL);

    /* Check model files exist */
    if (access(encoder, F_OK) != 0) {
        fprintf(stderr, "ERROR: Model not found at %s\nRun setup.sh first.\n", encoder);
        return 1;
    }
    if (access(vad_model, F_OK) != 0) {
        fprintf(stderr, "ERROR: Silero VAD not found at %s\nRun setup.sh first.\n", vad_model);
        return 1;
    }

    /* Init X11 for text injection */
    x11_init();

    /* ── Load recognizer ────────────────────────────────────────────────── */
    printf("Loading Parakeet-TDT 0.6B v3 int8...\n");

    SherpaOnnxOfflineRecognizerConfig config;
    memset(&config, 0, sizeof(config));
    config.model_config.transducer.encoder = encoder;
    config.model_config.transducer.decoder = decoder;
    config.model_config.transducer.joiner  = joiner;
    config.model_config.tokens             = tokens;
    config.model_config.num_threads        = NUM_THREADS;
    config.model_config.model_type         = "nemo_transducer";
    config.feat_config.sample_rate         = SAMPLE_RATE;
    config.feat_config.feature_dim         = 80;
    config.decoding_method                 = "greedy_search";

    g_recognizer = SherpaOnnxCreateOfflineRecognizer(&config);
    if (!g_recognizer) {
        fprintf(stderr, "ERROR: Failed to create recognizer.\n");
        return 1;
    }
    printf("  Recognizer loaded.\n");

    /* ── Load VAD ───────────────────────────────────────────────────────── */
    printf("Loading Silero VAD...\n");

    SherpaOnnxVadModelConfig vad_config;
    memset(&vad_config, 0, sizeof(vad_config));
    vad_config.silero_vad.model                = vad_model;
    vad_config.silero_vad.threshold            = VAD_THRESHOLD;
    vad_config.silero_vad.min_silence_duration = VAD_MIN_SILENCE;
    vad_config.silero_vad.min_speech_duration  = VAD_MIN_SPEECH;
    vad_config.silero_vad.max_speech_duration  = VAD_MAX_SPEECH;
    vad_config.silero_vad.window_size          = VAD_WINDOW_SIZE;
    vad_config.sample_rate                     = SAMPLE_RATE;

    g_vad = SherpaOnnxCreateVoiceActivityDetector(&vad_config, 60.0f);
    if (!g_vad) {
        fprintf(stderr, "ERROR: Failed to create VAD.\n");
        SherpaOnnxDestroyOfflineRecognizer(g_recognizer);
        return 1;
    }
    printf("  VAD loaded.\n");

    /* ── Init queue and worker thread ───────────────────────────────────── */
    queue_init(&g_queue);

    pthread_t worker;
    pthread_create(&worker, NULL, transcription_worker, NULL);

    /* ── Init PortAudio ─────────────────────────────────────────────────── */
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        fprintf(stderr, "ERROR: PortAudio init failed: %s\n", Pa_GetErrorText(err));
        return 1;
    }

    PaStream *stream = NULL;
    err = Pa_OpenDefaultStream(&stream, 1, 0, paFloat32, SAMPLE_RATE,
                               VAD_WINDOW_SIZE, pa_callback, NULL);
    if (err != paNoError) {
        fprintf(stderr, "ERROR: Pa_OpenDefaultStream failed: %s\n", Pa_GetErrorText(err));
        Pa_Terminate();
        return 1;
    }

    err = Pa_StartStream(stream);
    if (err != paNoError) {
        fprintf(stderr, "ERROR: Pa_StartStream failed: %s\n", Pa_GetErrorText(err));
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    /* ── Running ────────────────────────────────────────────────────────── */
    printf("\n");
    printf("==================================================\n");
    printf("  DICTATION ACTIVE — just speak!\n");
    printf("  Model: Parakeet-TDT 0.6B v3 int8 (CPU)\n");
    printf("  Ctrl+C to quit\n");
    printf("==================================================\n");
    printf("\n  Listening...\n\n");

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    while (g_running) {
        Pa_Sleep(100);
    }

    /* ── Cleanup ────────────────────────────────────────────────────────── */
    printf("\nShutting down...\n");

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    /* Wake up worker thread so it can exit */
    pthread_mutex_lock(&g_queue.mutex);
    pthread_cond_signal(&g_queue.cond);
    pthread_mutex_unlock(&g_queue.mutex);
    pthread_join(worker, NULL);

    queue_destroy(&g_queue);
    SherpaOnnxDestroyVoiceActivityDetector(g_vad);
    SherpaOnnxDestroyOfflineRecognizer(g_recognizer);

    x11_cleanup();
    printf("Bye!\n");
    return 0;
}
