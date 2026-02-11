CC = gcc
CFLAGS = -O2 -Wall -Wextra -Ilibs/include
LDFLAGS = -Llibs/lib -Wl,-rpath,'$$ORIGIN/libs/lib'
LIBS = -lsherpa-onnx-c-api -lportaudio -lpthread -lxkbcommon

dictate: dictate.c typer.c typer.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ dictate.c typer.c $(LIBS)

clean:
	rm -f dictate

.PHONY: clean
