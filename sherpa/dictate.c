/*
 * dictate.c — Sherpa-ONNX dictation in C.
 *
 * Single-file, zero Python dependencies. Links against libsherpa-onnx-c-api
 * and libportaudio. Uses Parakeet-TDT 0.6B v3 int8 + Silero VAD.
 * Text injection via /dev/uinput + xkbcommon (layout-aware, any compositor).
 *
 * Build:  make
 * Run:    ./dictate
 */

#include <ctype.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <linux/input.h>
#include <portaudio.h>
#include <sherpa-onnx/c-api/c-api.h>
#include "typer.h"

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

/* Path prefix for models (set from argv[0] location) */
static char g_basedir[4096];

/* ── Voice commands (trailing phrase → key press) ─────────────────────────── */

typedef struct {
    const char *phrase;    /* lowercase trailing phrase to match */
    int         keycode;   /* linux evdev keycode */
    int         ctrl;      /* hold Ctrl? */
    const char *label;     /* for logging */
} VoiceCommand;

static const VoiceCommand g_commands[] = {
    { "press enter", KEY_ENTER, 0, "Enter"  },
    { "press tab",   KEY_TAB,   0, "Tab"    },
    { "interrupt it", KEY_C,     1, "Ctrl+C" },
    { "cancel it",    KEY_C,     1, "Ctrl+C" },
    { NULL, 0, 0, NULL }
};

/* Check if text ends with a voice command (case-insensitive).
 * Returns the command, or NULL. Sets *cmd_start to where the phrase begins. */
static const VoiceCommand *match_trailing_command(const char *text, int len, int *cmd_start) {
    for (const VoiceCommand *cmd = g_commands; cmd->phrase; cmd++) {
        int plen = (int)strlen(cmd->phrase);
        if (plen > len) continue;

        const char *suffix = text + len - plen;

        /* Case-insensitive compare */
        int match = 1;
        for (int i = 0; i < plen; i++) {
            if (tolower((unsigned char)suffix[i]) != cmd->phrase[i]) {
                match = 0;
                break;
            }
        }
        if (!match) continue;

        /* Must be at word boundary (start of string or preceded by space) */
        if (suffix > text && suffix[-1] != ' ') continue;

        *cmd_start = (int)(suffix - text);
        return cmd;
    }
    return NULL;
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
        if (isalnum(c) || c >= 0xC0) has_alnum = 1; /* ASCII alnum or UTF-8 lead byte */
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

            int tlen = (int)strlen(text);
            /* Trim trailing whitespace and punctuation */
            while (tlen > 0 && (isspace((unsigned char)text[tlen - 1])
                   || text[tlen - 1] == '.' || text[tlen - 1] == ','
                   || text[tlen - 1] == '!' || text[tlen - 1] == '?'))
                tlen--;

            /* Check for trailing voice command */
            int cmd_start = tlen;
            const VoiceCommand *cmd = match_trailing_command(text, tlen, &cmd_start);

            /* Strip trailing space before the command */
            int text_len = cmd_start;
            while (text_len > 0 && text[text_len - 1] == ' ') text_len--;

            double t2 = now_ms();

            /* Type the text portion (if any) */
            if (text_len > 0) {
                char *buf = (char *)malloc(text_len + 2);
                if (buf) {
                    memcpy(buf, text, text_len);
                    buf[text_len] = ' ';
                    buf[text_len + 1] = '\0';
                    typer_type(buf);
                    free(buf);
                }
            }

            /* Fire the key command (if matched) */
            if (cmd) {
                typer_press(cmd->keycode, cmd->ctrl);
            }

            double t3 = now_ms();

            /* Log */
            if (cmd && text_len > 0) {
                fprintf(stderr, "\r\033[K  >> %.*s  [%s]\n", text_len, text, cmd->label);
            } else if (cmd) {
                fprintf(stderr, "\r\033[K  >> [%s]\n", cmd->label);
            } else {
                fprintf(stderr, "\r\033[K  >> %.*s\n", tlen, text);
            }
            fprintf(stderr, "     [%.1fs audio | recognize: %.0fms | type: %.0fms | total: %.0fms]\n",
                    duration, t1 - t0, t3 - t2, t3 - t0);
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

    /* Initialize text injection (uinput + xkbcommon) */
    if (typer_init() != 0)
        return 1;

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
    typer_cleanup();
    SherpaOnnxDestroyVoiceActivityDetector(g_vad);
    SherpaOnnxDestroyOfflineRecognizer(g_recognizer);

    printf("Bye!\n");
    return 0;
}
