{
  lib,
  sourceFiles ? lib.fileset.unions [./broquant ./pyproject.toml],
  buildPythonPackage,

  setuptools-scm,

  torch,
  torchvision,
  more-itertools,
}:
let
  fs = lib.fileset;
in

buildPythonPackage rec {
  pname = "broquant";
  version = "0.0.1";
  pyproject = true;

  src = fs.toSource {
    root = ./.;
    fileset = sourceFiles;
  };

  build-system = [
    setuptools-scm
  ];

  dependencies = [
    torch
    torchvision
    more-itertools
  ];

  meta = {
    description = "An example of neural network quantization from scratch.";
    homepage = "https://github.com/BROsandr/quantization_example";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ brosandr ];
  };
}
