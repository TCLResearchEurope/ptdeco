## TODO

+ **[general]** Implement and test building a wheel package
+ **[general]** Specify device via config
+ **[general]** Check device switching handling (cpu decomposition, gpu trainining etc.)
+ **[fal]** Add tests
+ **[fal]** Add decomposition params handling through config
+ **[trainable]** Refactor - split ptdeco.py into multiple modules
+ **[trainable]** Idea - add some options to customize decomposition process (e.g, type of deconvolution decomposition)

## DONE

+ **[general]** Add checking if all blacklisted modules are present in the model
+ **[general]** Clean-up configurator getters to make use of pydantic models
+ **[general]** Switch from dict to pydantic models
+ **[trainable]** Fix trainable decomposition script
+ **[general]** Add pydantic validators to trainer
+ **[general]** Refactor replace submodule in place
+ **[trainable]** Add per module metadata in decompose config
+ **[fal]** Add per module metadata in decompose config
+ **[direct]** Add creation of decompose config
+ **[direct]** Add reporting flops/params before and after decomposition
