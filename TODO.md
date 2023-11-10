## TODO

+ **[falor]** Add tests
+ **[general]** Implement and test building a wheel package
+ **[general]** Add logging decomposed module statistics by type in apply decompose config
+ **[lockd]** Idea - add some options to customize decomposition process (e.g, type of deconvolution decomposition)
+ **[trainer]** Add requirements.txt
+ **[trainer]** Check device switching handling (cpu decomposition, gpu trainining etc.)
+ **[trainer]** Specify device via config
+ **[trainer]** Move aux notebooks to a sepearte dir

## DONE

## ptdeco trainer 0.7.0
+ **[tranier]** Add logging state dict stats
+ **[trainer]** Fix state dict bug
+ **[trainer]** Unify logged stats of decomposed models in lockd/falor/finetune

## ptdeco 0.3.0, ptdeco trainer 0.6.0
+ **[falor]** Refactor `fal` -> `falor` and move module to separate directory
+ **[general]** Refactor wrapping class names
+ **[lockd]** Get rid of needless module copies in lockd wrapping
+ **[lockd]** Refactor - split ptdeco.py into multiple modules
+ **[lockd]** Remove `uwrap_in_place`
+ **[trainer]** Refactor compile config to Optional in trainer

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
