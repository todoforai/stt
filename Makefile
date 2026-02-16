CC = gcc
CFLAGS = -O2 -Wall -Wextra -Ilibs/include
LDFLAGS = -Llibs/lib -Wl,-rpath,'$$ORIGIN/libs/lib'
LIBS = -lsherpa-onnx-c-api -lportaudio -lpthread -lxkbcommon

all: dictate overlay

dictate: dictate.c typer.c typer.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ dictate.c typer.c $(LIBS)

overlay: overlay.c
	$(CC) -O2 -Wall -Wextra -o $@ overlay.c -lX11 -lXext -lXrender -lm

clean:
	rm -f dictate overlay

.PHONY: all clean
