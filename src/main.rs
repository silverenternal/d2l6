use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use tch::nn::{self, Module, ModuleT, OptimizerConfig, SequentialT};
use tch::vision::image;
use tch::{Cuda, Device, Kind, Tensor};

const SCENE_CLASSES: [&str; 6] = ["buildings", "forest", "glacier", "mountain", "sea", "street"];
const IMAGE_SIZE: i64 = 96;
const TRAIN_PER_CLASS: usize = 800;
const TEST_PER_CLASS: usize = 200;
const BATCH_SIZE: i64 = 64;
const EPOCHS: i64 = 12;

#[derive(Debug)]
struct ParallelModule<N1, N2> {
    net1: N1,
    net2: N2,
}

impl<N1, N2> ParallelModule<N1, N2> {
    fn new(net1: N1, net2: N2) -> Self {
        Self { net1, net2 }
    }
}

impl<N1, N2> Module for ParallelModule<N1, N2>
where
    N1: Module,
    N2: Module,
{
    fn forward(&self, xs: &Tensor) -> Tensor {
        Tensor::cat(&[self.net1.forward(xs), self.net2.forward(xs)], 1)
    }
}

#[derive(Debug)]
struct SharedMlp {
    input: nn::Linear,
    shared: nn::Linear,
    output: nn::Linear,
}

impl SharedMlp {
    fn new(vs: &nn::Path<'_>, in_features: i64, hidden: i64, out_features: i64) -> Self {
        let input = nn::linear(vs / "input", in_features, hidden, Default::default());
        let shared = nn::linear(vs / "shared", hidden, hidden, Default::default());
        let output = nn::linear(vs / "output", hidden, out_features, Default::default());
        Self { input, shared, output }
    }

    fn parameter_report(&self) -> String {
        format!(
            "input.w mean={:.4}, shared.w mean={:.4}, output.w mean={:.4}",
            tensor_mean(&self.input.ws),
            tensor_mean(&self.shared.ws),
            tensor_mean(&self.output.ws),
        )
    }

    fn gradient_report(&self) -> String {
        format!(
            "input.w grad_norm={:.4}, shared.w grad_norm={:.4}, output.w grad_norm={:.4}",
            grad_norm(&self.input.ws),
            grad_norm(&self.shared.ws),
            grad_norm(&self.output.ws),
        )
    }
}

impl Module for SharedMlp {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let h1 = xs.apply(&self.input).relu();
        let h2 = h1.apply(&self.shared).relu();
        let h3 = h2.apply(&self.shared).relu();
        h3.apply(&self.output)
    }
}

fn tensor_mean(tensor: &Tensor) -> f64 {
    f64::try_from(tensor.mean(Kind::Float)).unwrap()
}

fn grad_norm(tensor: &Tensor) -> f64 {
    let grad = tensor.grad();
    if grad.defined() {
        f64::try_from(grad.norm()).unwrap()
    } else {
        0.0
    }
}

fn run_parallel_module_demo(device: Device) {
    println!("========== A1. ParallelModule 演示 ==========");

    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let net1 = nn::linear(&root / "net1", 4, 3, Default::default());
    let net2 = nn::linear(&root / "net2", 4, 2, Default::default());
    let parallel = ParallelModule::new(net1, net2);

    let x = Tensor::randn([2, 4], (Kind::Float, device));
    let y = parallel.forward(&x);

    println!("随机输入 x shape = {:?}", x.size());
    x.print();
    println!("拼接输出 y shape = {:?}", y.size());
    y.print();
}

fn run_shared_parameter_mlp(device: Device) -> Result<()> {
    println!("\n========== A2. 共享参数 MLP 训练 ==========");

    let vs = nn::VarStore::new(device);
    let model = SharedMlp::new(&vs.root(), 4, 8, 1);
    let mut optimizer = nn::Sgd::default().build(&vs, 0.05)?;

    let n = 128;
    let x = Tensor::randn([n, 4], (Kind::Float, device));
    let true_w = Tensor::from_slice(&[1.5_f32, -2.0, 0.7, 3.0]).reshape([4, 1]).to_device(device);
    let noise = Tensor::randn([n, 1], (Kind::Float, device)) * 0.05;
    let y = x.matmul(&true_w) + 0.3 + noise;

    for epoch in 1..=8 {
        optimizer.zero_grad();

        let pred = model.forward(&x);
        let loss = pred.mse_loss(&y, tch::Reduction::Mean);
        loss.backward();

        println!(
            "epoch {:02}: loss={:.6}; {}; {}",
            epoch,
            f64::try_from(&loss).unwrap(),
            model.parameter_report(),
            model.gradient_report(),
        );

        optimizer.step();
    }

    println!("\n训练后参数观察: {}", model.parameter_report());
    println!("说明: `shared` 层在 forward 中被调用两次，因此它只有一组参数，但梯度会累计两次使用产生的贡献。");

    Ok(())
}

fn run_calculation_answers() {
    println!("========== 1-3. 计算题 ==========");

    let mlp_params = 3 * 4 + 4 + 4 * 1 + 1;
    println!("3.1 三层全连接网络参数量 = 3*4 + 4 + 4*1 + 1 = {mlp_params}");

    let conv_out = ((11 + 2 * 0 - 5) / 2) + 1;
    let conv_params = (5 * 5 * 3 + 1) * 10;
    println!("3.2 卷积层输出 = {conv_out} x {conv_out} x 10, 参数量 = (5*5*3 + 1)*10 = {conv_params}");

    let pool_out = ((11 - 2) / 1) + 1;
    println!("3.3 平均池化输出 = {pool_out} x {pool_out} x 3, 参数量 = 0");
}

fn conv_bn_relu(vs: &nn::Path<'_>, c_in: i64, c_out: i64) -> SequentialT {
    let conv_config = nn::ConvConfig { padding: 1, bias: false, ..Default::default() };
    nn::seq_t()
        .add(nn::conv2d(vs / "conv", c_in, c_out, 3, conv_config))
        .add(nn::batch_norm2d(vs / "bn", c_out, Default::default()))
        .add_fn(|xs| xs.relu())
}

fn conv_block(vs: &nn::Path<'_>, c_in: i64, c_out: i64) -> SequentialT {
    nn::seq_t()
        .add(conv_bn_relu(&(vs / "conv1"), c_in, c_out))
        .add(conv_bn_relu(&(vs / "conv2"), c_out, c_out))
        .add_fn(|xs| xs.max_pool2d_default(2))
}

fn scene_cnn(vs: &nn::Path<'_>) -> SequentialT {
    nn::seq_t()
        .add(conv_block(&(vs / "block1"), 3, 32))
        .add(conv_block(&(vs / "block2"), 32, 64))
        .add(conv_block(&(vs / "block3"), 64, 128))
        .add(conv_block(&(vs / "block4"), 128, 256))
        .add_fn(|xs| xs.adaptive_avg_pool2d([1, 1]).flat_view())
        .add_fn_t(|xs, train| xs.dropout(0.4, train))
        .add(nn::linear(vs / "head", 256, SCENE_CLASSES.len() as i64, Default::default()))
}

fn image_files(dir: &Path, limit: usize) -> Result<Vec<PathBuf>> {
    let mut files = fs::read_dir(dir)
        .with_context(|| format!("读取图片目录失败: {}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    files.sort();
    files.truncate(limit);
    Ok(files)
}

fn load_scene_split(root: &Path, per_class: usize) -> Result<(Tensor, Tensor)> {
    let mut images = Vec::new();
    let mut labels = Vec::new();

    for (label, class_name) in SCENE_CLASSES.iter().enumerate() {
        let class_dir = root.join(class_name);
        let files = image_files(&class_dir, per_class)?;
        println!("加载 {:>9}: {} 张 ({})", class_name, files.len(), class_dir.display());
        for file in files {
            let img = image::load_and_resize(&file, IMAGE_SIZE, IMAGE_SIZE)
                .with_context(|| format!("读取图片失败: {}", file.display()))?;
            images.push(img.to_kind(Kind::Float) / 255.0);
            labels.push(label as i64);
        }
    }

    let images = Tensor::stack(&images, 0);
    let labels = Tensor::from_slice(&labels);
    Ok((images, labels))
}

fn evaluate_accuracy(net: &impl ModuleT, images: &Tensor, labels: &Tensor, device: Device) -> f64 {
    let mut correct = 0.0;
    let mut total = 0.0;
    for (batch_images, batch_labels) in tch::data::Iter2::new(images, labels, BATCH_SIZE)
        .return_smaller_last_batch()
        .to_device(device)
    {
        let logits = net.forward_t(&batch_images, false);
        let batch_correct = logits
            .argmax(-1, false)
            .eq_tensor(&batch_labels)
            .to_kind(Kind::Float)
            .sum(Kind::Float);
        correct += f64::try_from(batch_correct).unwrap();
        total += batch_labels.size()[0] as f64;
    }
    correct / total
}

fn run_scene_classification(device: Device) -> Result<()> {
    println!("\n========== 4. 场景分类 CNN ==========");

    let train_root = Path::new("data/intel-scenes/seg_train/seg_train");
    let test_root = Path::new("data/intel-scenes/seg_test/seg_test");
    if !train_root.exists() || !test_root.exists() {
        println!("未找到场景分类数据集，请先运行: bash scripts/download_intel_scenes.sh");
        return Ok(());
    }

    println!("使用设备: {:?}, CUDA/ROCm 可用: {}, 设备数: {}", device, Cuda::is_available(), Cuda::device_count());
    if device.is_cuda() {
        Cuda::cudnn_set_benchmark(true);
    }
    println!("数据集: Intel Image Classification, 类别数={}, 图片尺寸={}x{}", SCENE_CLASSES.len(), IMAGE_SIZE, IMAGE_SIZE);
    let (train_images, train_labels) = load_scene_split(train_root, TRAIN_PER_CLASS)?;
    let (test_images, test_labels) = load_scene_split(test_root, TEST_PER_CLASS)?;

    println!(
        "训练样本: {}, 测试样本: {}",
        train_images.size()[0],
        test_images.size()[0]
    );

    let vs = nn::VarStore::new(device);
    let net = scene_cnn(&vs.root());
    let params: usize = vs.trainable_variables().iter().map(|t| t.numel()).sum();
    println!("CNN 结构: Conv-BN-ReLU-Pool x4 -> AdaptiveAvgPool -> Dropout -> Linear");
    println!("可训练参数量: {params}");

    let mut optimizer = nn::AdamW::default().build(&vs, 1e-3)?;
    for epoch in 1..=EPOCHS {
        let mut total_loss = 0.0;
        let mut batches = 0;
        for (batch_images, batch_labels) in tch::data::Iter2::new(&train_images, &train_labels, BATCH_SIZE)
            .shuffle()
            .to_device(device)
        {
            let batch_images = tch::vision::dataset::augmentation(&batch_images, true, 4, 0);
            let logits = net.forward_t(&batch_images, true);
            let loss = logits.cross_entropy_for_logits(&batch_labels);
            optimizer.backward_step(&loss);
            total_loss += f64::try_from(loss).unwrap();
            batches += 1;
        }

        let train_acc = evaluate_accuracy(&net, &train_images, &train_labels, device);
        let test_acc = evaluate_accuracy(&net, &test_images, &test_labels, device);
        println!(
            "epoch {epoch:02}: loss={:.4}, train_acc={:.2}%, test_acc={:.2}%",
            total_loss / batches as f64,
            train_acc * 100.0,
            test_acc * 100.0
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    tch::manual_seed(42);
    let device = Device::cuda_if_available();

    run_parallel_module_demo(device);
    run_shared_parameter_mlp(device)?;
    run_calculation_answers();
    run_scene_classification(device)?;

    Ok(())
}
