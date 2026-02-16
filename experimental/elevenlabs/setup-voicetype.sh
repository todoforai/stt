#!/bin/bash
# Setup script for voicetype - Realtime voice to text typing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICETYPE_SCRIPT="$SCRIPT_DIR/voicetype"
CONFIG_DIR="$HOME/.config/voicetype"
CONFIG_FILE="$CONFIG_DIR/config.json"

echo "=== voicetype setup ==="
echo

# Check if voicetype script exists
if [[ ! -f "$VOICETYPE_SCRIPT" ]]; then
    echo "Error: voicetype script not found at $VOICETYPE_SCRIPT"
    exit 1
fi

# Detect display server
if [[ "$XDG_SESSION_TYPE" == "wayland" ]]; then
    DISPLAY_SERVER="wayland"
    TYPE_TOOL="wtype"
    CLIP_TOOL="wl-clipboard"
else
    DISPLAY_SERVER="x11"
    TYPE_TOOL="xdotool"
    CLIP_TOOL="xclip"
fi
echo "Detected display server: $DISPLAY_SERVER"

# Install Python dependencies
echo
echo "Installing Python dependencies..."
pip install --user websockets sounddevice numpy

# Install system dependencies
echo
echo "Installing system dependencies..."
if command -v apt &> /dev/null; then
    sudo apt update
    sudo apt install -y $TYPE_TOOL $CLIP_TOOL yad libnotify-bin
elif command -v dnf &> /dev/null; then
    sudo dnf install -y $TYPE_TOOL $CLIP_TOOL yad libnotify
elif command -v pacman &> /dev/null; then
    sudo pacman -S --noconfirm $TYPE_TOOL $CLIP_TOOL yad libnotify
else
    echo "Warning: Could not detect package manager. Please install manually:"
    echo "  - $TYPE_TOOL"
    echo "  - $CLIP_TOOL"
    echo "  - yad"
    echo "  - libnotify (notify-send)"
fi

# Make script executable
echo
echo "Making voicetype executable..."
chmod +x "$VOICETYPE_SCRIPT"

# Create symlink
echo
echo "Creating symlink in /usr/local/bin..."
sudo ln -sf "$VOICETYPE_SCRIPT" /usr/local/bin/voicetype

# Create config directory and file
echo
echo "Setting up configuration..."
mkdir -p "$CONFIG_DIR"

if [[ ! -f "$CONFIG_FILE" ]]; then
    cat > "$CONFIG_FILE" << 'EOF'
{
  "api_key": "",
  "language": ""
}
EOF
    echo "Created config file at $CONFIG_FILE"
else
    echo "Config file already exists at $CONFIG_FILE"
fi

# Prompt for API key if not set
CURRENT_KEY=$(grep -o '"api_key": *"[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4)
if [[ -z "$CURRENT_KEY" && -z "$ELEVENLABS_API_KEY" ]]; then
    echo
    read -p "Enter your ElevenLabs API key (or press Enter to skip): " API_KEY
    if [[ -n "$API_KEY" ]]; then
        sed -i "s/\"api_key\": *\"[^\"]*\"/\"api_key\": \"$API_KEY\"/" "$CONFIG_FILE"
        echo "API key saved to config."
    else
        echo "Skipped. Set your API key later in $CONFIG_FILE"
    fi
fi

# Setup keyboard shortcut (GNOME only)
echo
read -p "Set keyboard shortcut? Enter a letter (e.g. 'v' for Ctrl+V), or press Enter to skip: " SHORTCUT_KEY
if [[ -n "$SHORTCUT_KEY" ]]; then
    SHORTCUT_KEY="${SHORTCUT_KEY,,}"  # lowercase
    BINDING="<Control>${SHORTCUT_KEY}"

    # Find available slot
    for i in {0..9}; do
        NAME=$(gsettings get org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom$i/ name 2>/dev/null || echo "")
        if [[ -z "$NAME" || "$NAME" == "''" || "$NAME" == "'voicetype'" ]]; then
            SLOT="custom$i"
            break
        fi
    done

    if [[ -n "$SLOT" ]]; then
        PATH="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/$SLOT/"
        EXISTING=$(gsettings get org.gnome.settings-daemon.plugins.media-keys custom-keybindings 2>/dev/null)
        if [[ "$EXISTING" != *"$PATH"* ]]; then
            if [[ "$EXISTING" == "@as []" || "$EXISTING" == "[]" ]]; then
                gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings "['$PATH']"
            else
                gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings "${EXISTING%]*}, '$PATH']"
            fi
        fi
        gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$PATH name "voicetype"
        gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$PATH command "voicetype --toggle"
        gsettings set org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:$PATH binding "$BINDING"
        echo "Shortcut set: Ctrl+${SHORTCUT_KEY^^}"
    else
        echo "Could not find available shortcut slot"
    fi
fi

echo
echo "=== Setup complete ==="
echo
echo "Usage:"
echo "  voicetype          # Toggle listening"
echo "  voicetype --stop   # Stop listening"
echo "  voicetype --quit   # Quit service"
echo
echo "Config: $CONFIG_FILE"
