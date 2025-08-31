#!/bin/bash

# Script to open setup.sh in a new terminal window
# This handles different terminal emulators and operating systems gracefully

set -e

echo "🚀 Opening Databricks App Template Setup..."

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to detect the operating system
get_os() {
    case "$(uname -s)" in
        Darwin*) echo "macOS" ;;
        Linux*) echo "Linux" ;;
        MINGW*|MSYS*|CYGWIN*) echo "Windows" ;;
        *) echo "Unknown" ;;
    esac
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if an macOS app exists
app_exists_macos() {
    [ -d "/Applications/$1.app" ] || [ -d "$HOME/Applications/$1.app" ]
}

OS=$(get_os)

# Create a temporary file to signal when setup is complete (for auto-close functionality)
TEMP_FILE=$(mktemp)

case "$OS" in
    macOS)
        # Try different terminal emulators in order of preference
        if app_exists_macos "iTerm"; then
            echo "📱 Opening in iTerm..."
            osascript <<EOF
tell application "iTerm"
    create window with default profile
    tell current session of current window
        write text "cd '$SCRIPT_DIR' && ./setup.sh --auto-close '$TEMP_FILE' && exit"
    end tell
end tell
EOF
        elif app_exists_macos "Warp"; then
            echo "📱 Opening in Warp..."
            # Warp doesn't have great AppleScript support, so we use open command
            open -a Warp "$SCRIPT_DIR"
            echo "⚠️  Please run './setup.sh' in the Warp window that just opened"
            
        elif app_exists_macos "Terminal"; then
            echo "📱 Opening in Terminal..."
            osascript <<EOF
tell application "Terminal"
    do script "cd '$SCRIPT_DIR' && ./setup.sh --auto-close '$TEMP_FILE' && exit"
    activate
end tell
EOF
        else
            echo "❌ No supported terminal emulator found on macOS"
            echo "📋 Supported terminals: iTerm, Terminal, Warp"
            echo ""
            echo "💡 Please open a terminal manually and run:"
            echo "   cd '$SCRIPT_DIR'"
            echo "   ./setup.sh"
            exit 1
        fi
        ;;
        
    Linux)
        # Try different terminal emulators common on Linux
        if command_exists gnome-terminal; then
            echo "📱 Opening in GNOME Terminal..."
            gnome-terminal -- bash -c "cd '$SCRIPT_DIR' && ./setup.sh; exec bash"
            
        elif command_exists konsole; then
            echo "📱 Opening in Konsole..."
            konsole -e bash -c "cd '$SCRIPT_DIR' && ./setup.sh; exec bash"
            
        elif command_exists xfce4-terminal; then
            echo "📱 Opening in XFCE Terminal..."
            xfce4-terminal -e "bash -c 'cd $SCRIPT_DIR && ./setup.sh; exec bash'"
            
        elif command_exists xterm; then
            echo "📱 Opening in XTerm..."
            xterm -e "cd '$SCRIPT_DIR' && ./setup.sh; bash"
            
        else
            echo "❌ No supported terminal emulator found on Linux"
            echo "📋 Supported terminals: gnome-terminal, konsole, xfce4-terminal, xterm"
            echo ""
            echo "💡 Please open a terminal manually and run:"
            echo "   cd '$SCRIPT_DIR'"
            echo "   ./setup.sh"
            exit 1
        fi
        ;;
        
    Windows)
        echo "📱 Opening in Windows Terminal or Command Prompt..."
        # For Windows, we'll use start command
        if command_exists cmd.exe; then
            cmd.exe /c "start cmd /k cd /d $SCRIPT_DIR && bash setup.sh"
        else
            echo "❌ Could not open terminal on Windows"
            echo ""
            echo "💡 Please open a terminal manually and run:"
            echo "   cd '$SCRIPT_DIR'"
            echo "   ./setup.sh"
            exit 1
        fi
        ;;
        
    *)
        echo "❌ Unsupported operating system: $(uname -s)"
        echo ""
        echo "💡 Please open a terminal manually and run:"
        echo "   cd '$SCRIPT_DIR'"
        echo "   ./setup.sh"
        exit 1
        ;;
esac

# For macOS terminals that support auto-close, wait for completion
if [ "$OS" = "macOS" ] && { app_exists_macos "iTerm" || app_exists_macos "Terminal"; }; then
    echo "⏳ Waiting for setup to complete..."
    
    # Wait for the setup script to signal completion
    while [ ! -f "$TEMP_FILE" ] || [ "$(cat "$TEMP_FILE" 2>/dev/null)" != "setup_complete" ]; do
        sleep 1
    done
    
    # Clean up
    rm -f "$TEMP_FILE"
    echo "✅ Setup completed!"
fi