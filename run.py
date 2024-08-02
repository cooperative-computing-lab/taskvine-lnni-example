def run(weight_path, x, num_inferences_per_task):
    # Limit usable cores of tensorflow
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.run_functions_eagerly(True)
    
    from tensorflow.keras.applications.resnet50 import ResNet50
    model = ResNet50(weights=weight_path) 

    for i in range(num_inferences_per_task):
        preds = model.predict(x)
    return 1


def run_serverless(x, num_inferences_per_task):
    global model
    for i in range(num_inferences_per_task):
        preds = model.predict(x)
    return 1


def context_setup(args):
    # Limit usable cores of tensorflow
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.run_functions_eagerly(True)
    
    from tensorflow.keras.applications.resnet50 import ResNet50
    model = ResNet50(weights=args['weight_path'])
    return {'model': model, 'direct_mode': True}


def main():
    import time
    start = time.time()
    num_inferences_per_task = 16
    num_tasks = 10
    env_tarball = 'env.tar.gz'
    weight_path = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    init_command = 'conda run -n lnni '
    cores_per_task = 2  # each worker has 4 cores as defined in run_worker.sh
    port = 9126

    # preprocess input data
    from tensorflow.keras.preprocessing import image
    import numpy as np
    from tensorflow.keras.applications.resnet50 import preprocess_input
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # get run mode
    import sys
    mode = sys.argv[1]

    if mode == 'local-p':
        run(weight_path, x, num_inferences_per_task)
        exit(1)
    elif mode == 'local-s':
        d = context_setup({'weight_path': weight_path})
        for k in d:
            globals()[k] = d[k]
        run_serverless(x, num_inferences_per_task)
        exit(2)
    else:    
        import ndcctools.taskvine as vine
        q = vine.Manager(port=port, name='nn_exp')
        print(f"TaskVine manager listening on port {q.port}")
        weight_path_vine_file = q.declare_file(weight_path, cache=True, peer_transfer=True)
        
        if mode == 'remote-p':
            env_tarball_vine_file = q.declare_poncho(env_tarball, cache=True, peer_transfer=True)
            for i in range(num_tasks):
                t = vine.PythonTask(run, weight_path, x, num_inferences_per_task)
                t.add_environment(env_tarball_vine_file)
                t.add_input(weight_path_vine_file, weight_path)
                t.set_cores(cores_per_task)
                q.submit(t)
        elif mode == 'remote-r':
            for i in range(num_tasks):
                t = vine.PythonTask(run, weight_path, x, num_inferences_per_task)
                t.set_cores(cores_per_task)
                t.set_command(init_command + t.command)
                q.submit(t)
        elif mode == 'remote-s':
            print("Creating library from functions...")
            libtask = q.create_library_from_functions('lib', run_serverless, context=context_setup, context_arg={'weight_path': weight_path}, poncho_env='env.tar.gz')
            libtask.add_input(weight_path_vine_file, weight_path)
            libtask.set_cores(cores_per_task)
            q.install_library(libtask)
            
            print("Submitting function call tasks...")
            for i in range(num_tasks):
                t = vine.FunctionCall('lib', 'run_serverless', x, num_inferences_per_task)
                t.set_exec_method('direct')
                q.submit(t)
        else:
            raise Exception

    num_completed_tasks = 0
    while not q.empty():
        t = q.wait(5)
        if t:
            num_completed_tasks += t.output
            print(f'Task {num_completed_tasks}/{num_tasks} returned with output {t.output}')

    print(f'Completed: {num_completed_tasks}/{num_tasks}, mode: {mode}')
    print('Elapsed:', time.time() - start)

main()
