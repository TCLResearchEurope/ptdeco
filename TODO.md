## TODO

+ **[general]** Implement and test building a wheel package
+ **[trainer]** Specify device via config
+ **[trainer]** Check device switching handling (cpu decomposition, gpu trainining etc.)
+ **[trainer]** Refactor compile config to Optional in trainer
+ **[falor]** Add tests

+ **[trainable]** Refactor - split ptdeco.py into multiple modules
+ **[trainable]** Idea - add some options to customize decomposition process (e.g, type of deconvolution decomposition)

## DONE
+ **[falor]** Refactor `fal` -> `falor` and move module to separate directory

## ptdeco 0.2.0, ptdeco trainer 0.5.0
+ **[falor]** Add decomposition params handling through config
+ **[general]** Add checking if all blacklisted modules are present in the model
+ **[general]** Clean-up configurator getters to make use of pydantic models
+ **[general]** Switch from dict to pydantic models
+ **[trainable]** Fix trainable decomposition script
+ **[general]** Add pydantic validators to trainer
+ **[general]** Refactor replace submodule in place
+ **[trainable]** Add per module metadata in decompose config
+ **[falor]** Add per module metadata in decompose config
+ **[direct]** Add creation of decompose config
+ **[direct]** Add reporting flops/params before and after decomposition
