#!/bin/bash

# Exit on error
set -e

# Create build directory
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Create AppDir structure
APPDIR="SterlingX_BootStrap_Node.AppDir"
mkdir -p "$APPDIR/usr/bin"
mkdir -p "$APPDIR/usr/lib"
mkdir -p "$APPDIR/usr/share/applications"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"

# Copy application files
cp ../full_node.py "$APPDIR/usr/bin/sterlingx"
# Copy all Python modules
cp ../*.py "$APPDIR/usr/lib/"
# Copy icon to both locations
cp ../logo.png "$APPDIR/sterlingx.png"
cp ../logo.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/sterlingx.png"

# Create desktop file
cat > "$APPDIR/sterlingx.desktop" << EOF
[Desktop Entry]
Type=Application
Name=SterlingX Bootstrap Node
Comment=SterlingX Blockchain Bootstrap Node
Exec=sterlingx
Icon=sterlingx
Categories=Network;Finance;
Terminal=true
EOF

# Create AppRun script
cat > "$APPDIR/AppRun" << EOF
#!/bin/bash

# Set environment variables
export PYTHONPATH="\$APPDIR/usr/lib:\$APPDIR/usr/lib/python3.11/site-packages:\$PYTHONPATH"
export PATH="\$APPDIR/usr/bin:\$PATH"
export LD_LIBRARY_PATH="\$APPDIR/usr/lib:\$LD_LIBRARY_PATH"

# Run the application
exec "\$APPDIR/usr/bin/sterlingx" "\$@"
EOF
chmod +x "$APPDIR/AppRun"

# Create Python virtual environment in AppDir
python3.11 -m venv "$APPDIR/usr"
source "$APPDIR/usr/bin/activate"

# Install required packages
pip install --upgrade pip
pip install -r ../requirements.txt

# Download and extract appimagetool
wget https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x appimagetool-x86_64.AppImage

# Create AppImage
./appimagetool-x86_64.AppImage "$APPDIR" ../SterlingX_BootStrap_Node-x86_64.AppImage

# Cleanup
cd ..
rm -rf "$BUILD_DIR" 