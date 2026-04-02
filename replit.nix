{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.streamlit
    pkgs.opencv4
    pkgs.libGL
    pkgs.libGLU
  ];
}
