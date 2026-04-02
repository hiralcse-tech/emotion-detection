{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.virtualenv
    pkgs.opencv4
    pkgs.ffmpeg
    pkgs.stdenv.cc.cc.lib
    pkgs.libGL
    pkgs.libGLU
  ];
}
