## TODO
+ **[falor]** Add tests
+ **[general]** Implement and test building a wheel package
+ **[lockd]** Make wrapping true wrapping (instead of copying)
+ **[lockd]** Idea - add some options to customize decomposition process (e.g, type of deconvolution decomposition)
+ **[lockd]** Implement or remove uwrap_in_place
+ **[trainer]** Add requirements.txt
+ **[trainer]** Check device switching handling (cpu decomposition, gpu trainining etc.)
+ **[trainer]** Specify device via config

## DONE
+ **[general]** Refactor wrapping class names
+ **[lockd]** Refactor - split ptdeco.py into multiple modules
+ **[trainer]** Refactor compile config to Optional in trainer
+ **[falor]** Refactor `fal` -> `falor` and move module to separate directory

## ptdeco 0.2.0, ptdeco trainer 0.5.0
+ **[falor]** Add decomposition params handling through config
+ **[general]** Add checking if all blacklisted modules are present in the model
+ **[general]** Clean-up configurator getters to make use of pydantic models
+ **[general]** Switch from dict to pydantic models
+ **[lockd]** Fix lockd decomposition script
+ **[general]** Add pydantic validators to trainer
+ **[general]** Refactor replace submodule in place
+ **[lockd]** Add per module metadata in decompose config
+ **[falor]** Add per module metadata in decompose config
+ **[direct]** Add creation of decompose config
+ **[direct]** Add reporting flops/params before and after decomposition
