"""
Main file for training and loading models
Behaviour dictated by runtime arguments
Author: Robert Trew
"""
import os
import keras
from keras import optimizers
import code_util as util
import loss
import argparse


def main(
    data_class: util,
    model_class,
    continue_training: bool = True,
    change_lf: bool = True,
    **kwargs,
):
    history = dict()
    params = dict()
    loaded_model = data_class.loaded_model(model_class)
    if loaded_model:
        model = model_class
        data_class.precision = model.input.dtype.name
        input_shape = tuple(model.input.shape)
        try:
            history = data_class.load_pickled("history")
            params = data_class.load_pickled("params")
        except ValueError:
            # Continue with empty dictionaries
            pass
        try:
            data_class.c_space = params["colourspace"]
        except KeyError:
            # Continue with runtime arg
            pass
        if data_class.sequences:
            data_class.set_input_dims(input_shape[2:])
        else:
            data_class.set_input_dims(input_shape[1:])
    if not loaded_model or continue_training:
        input_dims = data_class.get_input_dims()
        if loaded_model:
            chosen_model = data_class.get_model_from_string(model.name)(
                input_dims, **kwargs
            )
        else:
            chosen_model = model_class(input_dims, **kwargs)
            model = chosen_model.build()

        kwargs.pop("c_space")
        # Setting learning rate from average of last model
        # history.get("lr", [0.0005])[-1] to get last learning rate
        # params.get("lr", 0.0005) to get average learning rate
        adam = optimizers.adam(
            learning_rate=history.get("lr", [5e-4])[-1],
            beta_1=0.9,
            beta_2=0.999,
            amsgrad=False,
        )  # Best so far
        # nadam = optimizers.nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
        # rms_prop = optimizers.rmsprop(learning_rate=0.001, rho=0.9)
        # sgd = optimizers.sgd(learning_rate=0.001, momentum=0.0, nesterov=False)

        if data_class.sequences:
            if loaded_model and change_lf:
                model.compile(
                    optimizer=adam,
                    loss=loss.tf_ms_ssim_vid,
                    metrics=[loss.tf_ms_ssim_vid, keras.losses.mse, loss.tf_psnr_vid],
                )
            else:
                model.compile(
                    optimizer=adam,
                    loss=keras.losses.mse,
                    metrics=[keras.losses.mse, loss.tf_psnr_vid, loss.tf_ms_ssim_vid],
                )
        else:
            model.compile(
                optimizer=adam,
                loss=loss.tf_ms_ssim,
                metrics=[loss.tf_ms_ssim, loss.tf_ssim, loss.tf_psnr, keras.losses.mse],
            )

        model.summary(print_fn=print)

        history = chosen_model.train(model, util_class=data_class, **kwargs)

        history.params.update({"batch_size": kwargs.get("batch_size", "unknown")})

        lr = history.history["lr"]
        avg_lr = float(round(sum(lr) / len(lr), ndigits=8))
        history.params.update({"lr": avg_lr})
        kwargs.update({"training_data": history})
    kwargs.update(
        {"loaded_model": loaded_model, "continue_training": continue_training}
    )
    avg_time = data_class.output_results(model, **kwargs)

    print(f"Average time to predict: {avg_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-ci",
        "--c-input",
        dest="c_input",
        help="Path to compressed input - for training",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-oi",
        "--o-input",
        dest="o_input",
        help="Path to original input - for training",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-od",
        "--out-dir",
        dest="o_dir",
        help="Path to store outputs - training data and results",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        help="Model to use - ClassName or path to pre-trained model",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-nt",
        "--no-train",
        dest="train",
        action="store_true",
        help="Only produce output from loaded model",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--sequences",
        dest="seq",
        action="store_true",
        help="Boolean - training sequences (video)",
        default=False,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        dest="run_epochs",
        help="Epochs to run model for",
        default=10,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        help="Batch size for model",
        default=2,
        type=int,
    )
    parser.add_argument(
        "-d",
        "--dims",
        dest="dims",
        help="Specified input dimensions (width, height, channels)",
        default="512, 768, 3",
        type=str,
    )
    parser.add_argument(
        "-cs",
        "--colour-space",
        dest="c_space",
        help="Specified colour-space, default: YUV",
        default="YUV",
        type=str,
    )
    parser.add_argument(
        "-kl",
        "--keep-lf",
        dest="keep_lf",
        help="Boolean to change loss function",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    data_man = util.DataManagement(
        script_dir=os.getcwd(),
        sequences=args.seq,
        c_data=args.c_input,
        o_data=args.o_input,
        out_dir=args.o_dir,
        input_dims=args.dims,
        c_space=args.c_space,
    )

    print(f"ci: {args.c_input}")
    print(f"oi: {args.o_input}")
    print(f"od: {args.o_dir}")
    print(f"m: {args.model}")
    print(f"t: {args.train}")
    print(f"s: {args.seq}")
    print(f"e: {args.run_epochs}")
    print(f"b: {args.batch_size}")
    print(f"d: {data_man.input_dims.get('dims', 'not set')}")
    print(f"cs: {args.c_space}")
    print(f"clf: {args.keep_lf}")

    try:
        use_model = data_man.get_model_from_string(args.model)
    except AttributeError:
        use_model = data_man.load_model_from_path(args.model)

    main(
        data_man,
        use_model,
        continue_training=args.train,
        change_lf=not args.keep_lf,
        **{
            "run_epochs": args.run_epochs,
            "batch_size": args.batch_size,
            "c_space": args.c_space,
        },
    )
