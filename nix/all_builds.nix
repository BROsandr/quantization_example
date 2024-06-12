{ sources ? import ./sources.nix }:
let
  pkgs = import sources.nixpkgs { config = {}; overlays = []; };
  defaultBuild = pkgs.callPackage ./.. { inherit (pkgs.python3.pkgs)
    setuptools-scm
    torch
    torchvision
    more-itertools
    buildPythonPackage;
  };

  shell = pkgs.mkShellNoCC {
    inputsFrom = [ defaultBuild ];
    packages = with pkgs; [ ];

    hardeningDisable = ["all"];
  };
in
{
  inherit defaultBuild shell;
}
