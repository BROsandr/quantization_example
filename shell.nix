{ sources ? import ./nix/sources.nix }:
let
  pkgs = import sources.nixpkgs { config = {}; overlays = []; };
  pyEnv = pkgs.python3.withPackages (ps: with ps; [
    torch
    torchvision
  ]);
  shell = pkgs.mkShellNoCC {
    inputsFrom = [ ];
    packages = with pkgs; [
      pyEnv
    ];

    hardeningDisable = ["all"];
  };
in {
  inherit shell;
}
