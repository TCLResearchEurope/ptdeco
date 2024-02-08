## TODO
+ **[general]** NSR loss mode?
+ **[falor]** Add tests
+ **[lockd]** Idea - add some options to customize decomposition process (e.g, type of deconvolution decomposition)
+ **[trainer]** Add requirements.txt
+ **[trainer]** Check device switching handling (cpu decomposition, gpu trainining etc.)
+ **[trainer]** Specify device via config

## DONE

## ptdeco 0.4.0, ptdeco trainer 0.8.0
+ **[general]** Implement and test building a wheel package
+ **[general]** Move to src project layout
+ **[utils]** Add logging num of decomposed modules of each type in apply_decompose_config_in_place
+ **[falor]** Fix nsr loss bug
+ **[lockd]** Improve typing in losses
+ **[trainer]** Add timming to every run
+ **[trainer]** Add saving final decompose config and state dict of finetuned model
+ **[trainer]** Fix typing
+ **[trainer]** Move notebooks to a sepearte dir

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
