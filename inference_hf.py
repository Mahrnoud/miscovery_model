from transformers import AutoModel, CONFIG_MAPPING, MODEL_MAPPING, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
import torch
from transformers import AutoTokenizer
import logging
import time
import numpy as np


# NLP evaluation metrics
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from bert_score import score

# Import custom model architecture 
from model_architecture import CustomTransformerConfig, CustomTransformerModel

# Language detection utilities
from langdetect import detect, LangDetectException

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===== MODEL REGISTRATION =====
# Register the custom model with Transformers library
def register_custom_model():
    """
    Register the custom transformer model with the Hugging Face transformers library
    so that it can be loaded with AutoModel and related classes.
    """
    # Register configuration with the configuration mapping
    CONFIG_MAPPING.register("miscovery", CustomTransformerConfig)

    # Register model with the model mapping
    MODEL_MAPPING.register(CustomTransformerConfig, CustomTransformerModel)

    # Register for sequence-to-sequence and causal language modeling tasks
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.register(CustomTransformerConfig, CustomTransformerModel)

    logger.info("Custom model successfully registered with transformers library")


# Register the model
register_custom_model()


# ===== MODEL LOADING =====
def load_model_and_tokenizer(model_name="miscovery/model"):
    """
    Load the model and tokenizer from the specified path or Hugging Face model hub.

    Args:
        model_name (str): Path or name of the model to load

    Returns:
        tuple: (tokenizer, model, device)
    """
    logger.info(f"Loading model and tokenizer from {model_name}")

    # Determine the device (GPU or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Move model to the appropriate device
        model = model.to(device)

        logger.info("Model and tokenizer loaded successfully")
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


# Load the model and tokenizer
tokenizer, model, device = load_model_and_tokenizer()


# ===== LANGUAGE DETECTION =====
def is_english(text):
    """
    Detect if the given text is in English.

    Args:
        text (str): Text to analyze

    Returns:
        bool: True if text is in English, False otherwise
    """
    try:
        # Remove language tags and take a sample of the text for efficiency
        sample_text = text[:500].replace("[LANG_EN]", "").replace("[LANG_AR]", "").strip()

        # If sample is empty after removing tags, fall back to character analysis
        if not sample_text:
            return any(ord(char) < 128 for char in text)

        # Use language detection library
        return detect(sample_text) == 'en'
    except LangDetectException:
        # Fallback method: count ASCII characters
        ascii_chars = sum(1 for char in text if ord(char) < 128)
        total_chars = len(text)

        if total_chars == 0:
            return True

        # If more than 50% characters are ASCII, likely English
        return (ascii_chars / total_chars) > 0.5


# ===== TEXT GENERATION =====
def generate_best_answer(model, prompt, tokenizer, device='cuda', max_length=128):
    """
    Optimized generation function for producing high-quality answers.

    Args:
        model: The transformer model
        prompt (str): Input text prompt
        tokenizer: Tokenizer for the model
        device (str): Device to use for inference ('cuda' or 'cpu')
        max_length (int): Maximum length of generated text

    Returns:
        str: Generated response
    """
    # Set model to evaluation mode
    model.eval()

    # Record generation start time
    start_time = time.time()

    try:
        # Configure generation parameters for best quality
        output = model.generate_response(
            prompt=prompt,
            tokenizer=tokenizer,
            max_length=max_length,
            device=device,
            temperature=0.4,  # Controls randomness (lower = more deterministic)
            top_k=25,  # Limits to top k tokens
            top_p=0.88,  # Nucleus sampling parameter
            repetition_penalty=1.15,  # Discourages token repetition
            do_sample=True  # Use sampling instead of greedy decoding
        )

        # Calculate and log generation time
        generation_time = time.time() - start_time
        logger.info(f"Text generated in {generation_time:.2f} seconds")

        return output
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        return ""


# ===== EVALUATION FUNCTIONS =====
def calculate_metrics(reference, generated, is_eng=None):
    """
    Calculate various NLP evaluation metrics between reference and generated text.

    Args:
        reference (str): Reference/ground truth text
        generated (str): Model-generated text
        is_eng (bool, optional): Whether text is in English (if None, will be detected)

    Returns:
        dict: Dictionary containing the metric scores
    """
    if is_eng is None:
        is_eng = is_english(reference)

    metrics = {}

    # Initialize ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Smoothing function for BLEU (helps with zero counts)
    smooth = SmoothingFunction().method1

    # Calculate BLEU score
    ref_tokens = [nltk.word_tokenize(reference.lower())]
    gen_tokens = nltk.word_tokenize(generated.lower())
    metrics['bleu'] = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smooth)

    # Calculate ROUGE scores
    rouge_scores = rouge.score(reference, generated)
    metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
    metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
    metrics['rougeL'] = rouge_scores['rougeL'].fmeasure

    # Calculate BERTScore
    # Use appropriate language model based on text language
    lang = "en" if is_eng else "ar"
    P, R, F1 = score([generated], [reference], lang=lang, model_type="bert-base-multilingual-cased")
    metrics['bertscore_f1'] = F1[0].item()

    return metrics


def display_evaluation_results(reference, generated, metrics):
    """
    Display the evaluation results in a readable format.

    Args:
        reference (str): Reference text
        generated (str): Generated text
        metrics (dict): Dictionary of calculated metrics
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print("\nREFERENCE:")
    print(reference.strip())

    print("\nGENERATED:")
    print(generated.strip())

    print("\nMETRICS:")
    print(f"BLEU score:    {metrics['bleu']:.4f}")
    print(f"ROUGE-1:       {metrics['rouge1']:.4f}")
    print(f"ROUGE-2:       {metrics['rouge2']:.4f}")
    print(f"ROUGE-L:       {metrics['rougeL']:.4f}")
    print(f"BERTScore F1:  {metrics['bertscore_f1']:.4f}")

    print("=" * 80 + "\n")


# ===== BATCH EVALUATION =====
def evaluate_batch(prompts, references):
    """
    Evaluate the model on a batch of prompts and references.

    Args:
        prompts (list): List of input prompts
        references (list): List of reference answers

    Returns:
        tuple: (generated_texts, average_metrics)
    """
    if len(prompts) != len(references):
        logger.error("Number of prompts and references must match")
        return [], {}

    generated_texts = []
    all_metrics = []

    logger.info(f"Starting batch evaluation of {len(prompts)} prompts")

    for i, (prompt, reference) in enumerate(zip(prompts, references)):
        logger.info(f"Processing prompt {i + 1}/{len(prompts)}")

        # Generate text
        generated = generate_best_answer(model, prompt, tokenizer, device)
        generated_texts.append(generated)

        # Calculate metrics
        is_eng = is_english(reference)
        metrics = calculate_metrics(reference, generated, is_eng)
        all_metrics.append(metrics)

        # Display individual results
        display_evaluation_results(reference, generated, metrics)

    # Calculate and return average metrics
    avg_metrics = {
        'bleu': np.mean([m['bleu'] for m in all_metrics]),
        'rouge1': np.mean([m['rouge1'] for m in all_metrics]),
        'rouge2': np.mean([m['rouge2'] for m in all_metrics]),
        'rougeL': np.mean([m['rougeL'] for m in all_metrics]),
        'bertscore_f1': np.mean([m['bertscore_f1'] for m in all_metrics]),
    }

    logger.info("Batch evaluation completed")
    return generated_texts, avg_metrics


# ===== MAIN EXECUTION =====
def main():
    """
    Main execution function to demonstrate the model evaluation.
    """
    # Download NLTK data if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer")
        nltk.download('punkt')

    # Example prompts for evaluation
    prompts = [
        "[LANG_EN] Summarize the following text: A man and woman have been killed in a light plane crash in Mudgee, two-hours north of Bathurst in New South Wales on Sunday. At about 10.55am emergency services were called to Mudgee Airport following reports of a light aircraft crash. It was reported that the plane had crashed into a paddock while on approach to land at the airport near Henry Lawson Drive. A man and woman were killed in a light plane crash in Mudgee, two-hours north of Bathurst in New South Wales . It is believed that the pilot and passenger were a couple. Officers from Mudgee Local Area Command attended the scene where they located a man, deceased inside the single-engine two-seater plane. The pilot who was in his 60s died on impact while the only passenger, a woman died on the scene as paramedics attempted to revive her. A crime scene has been established at the airport and will be examined. The  Australian Transport Safety Bureau and forensic officers joined a Police investigation. Stuart Godley from the Australian Transport Safety Bureau told ABC News that police will guard the site overnight and two investigators will arrive there tomorrow morning. 'We don't really know what's happened at this stage,' he said. 'All we know is that it is a small single engine aircraft, it's a Vans RV-6 and it's amateur built. 'That means it's been built by the owner, most likely the pilot.' A report will also be prepared for the information of the Coroner.",
        "[LANG_EN] In what year did the French Revolution begin?",
        "[LANG_EN] Translate this to Arabic: What year did World War I begin?",
        # """
        # [LANG_AR] قم بتلخيص النص التالي: أرسلت اليونان وبلغاريا قوات إضافية إلى الحدود المشتركة بين كل منهما وتركيا لمنع اللاجئين من عبورها "باصات لجوء مباشرة إلى أوروبا" وتداول المغردون صورا لـ "حافلات مجانية" تقوم بنقل اللاجئين السوريين من مدينة اسطنبول إلى مدينة أدرنة، الواقعة على الحدود التركية اليونانية. كما أظهرت مقاطع فيديو متداولة مشاهد تدافع أعداد كبيرة من الشباب السوربين لركوب الحافلة المتوجهة إلى الحدود اليونانية. وحسب مغردين، تقوم الشرطة التركية بتسهيل عبور اللاجئين للحدود، ولكن تبدأ الاشتباكات فور وصولهم إلى الأراضي الأوروبية، وهو ما جاء بعد التصعيد الكبير الذي شهده الشمال السوري والذي أسفر عن مقتل عشرات الجنود الأتراك. وكانت السلطات التركية قد أعلنت إنها لم تتلق دعماً كافياً في استقبالها لقرابة أربعة ملايين لاجئ سوري. مواضيع قد تهمك نهاية السوريون "سلعة تجارية" كما شارك بعض السوريين تجاربهم بعد الوصول إلى الحدود اليونانية عبر صفحات معنية بشؤون اللاجئين السوريين في تركيا، حيث وصف أحدهم تجربته بالـ "صعبة والمذلة" ونصح السوريين بالامتناع عن الهجرة لأنه يشعر "بالندم". وعبر عن استياءه من "اللعب بالسوريين كأنهم ليسوا بشرا". وانتقد يوسف الشريف سياسية حزب العدالة والتنمية التركي الحاكم بما يتعلق بالتعامل مع اللاجئين السوريين، مشيراً إلى "التناقض" بين الفعل والاتفاقيات. وأضاف قائلاً "من جديد يدفع اللاجئ السوري ثمن السياسات الداخلية للحزب الحاكم". وفي السياق نفسه، انتقد فريق من المغردين الرئيس التركي أردوغان مشيرين إلى تحول اللاجئين السوريين في الخطاب التركي من "ضيوف" للبلاد، إلى "سلعة جاهزة للتصدير". في المقابل، وعقب نبأ مقتل الجنود الأتراك، تضامن قطاع من المغردين مع تركيا والرئيس التركي تحت وسم #كلنا_مع تركيا و #تركيا_ليست_وحدها. وبرر آخرون قرارات الحكومة التركية بإلإشارة إلى الأزمة الاقتصادية التي تعاني منها تركيا والاتفاقيات مع الاتحاد الأوروبي التي "لم تتحقق بعد". كما أشار البعض إلى مسؤولية المجتمع الدولي أمام الكارثة الإنسانية التي يتعرض لها الشمال السوري. ودعا الأزهر إلى "تغليب مصلحة الشعب السوري الذي تحمَّل ما لا تتحمله الجبال والنظر بعين الإنسانية والرحمة إلى ما آلت إليه الأوضاع في إدلب ومدن سورية أخرى
        # """,
        "[LANG_EN]  Who invented the telephone?",
        "[LANG_EN] Translate this to Arabic: How can i play a game?",
#         """
# لخص النص التالي: يُنسب اختراع الهاتف إلى ألكسندر جراهام بيل. على الرغم من أن اختراعه لم يكن أول واحد من اخترع الهاتف قبل ذلك، إلا أن الفضل غالبًا ما يُنسب إليه الفضل في اختراع التليفون في عام 1876. ومع ذلك، فإن تاريخ الهاتف معقد، في عام 1936.
#         """,
#         "What is agile software development?",
#         "Summarize the following text: Agile software development is an iterative and incremental approach to software development that emphasizes flexibility, collaboration, and customer satisfaction. Unlike traditional waterfall models, Agile methods embrace changing requirements throughout the project lifecycle.  Key characteristics include short development cycles (sprints), frequent releases, continuous feedback, self-organizing teams, and a focus on delivering working software as quickly as possible.  Popular Agile frameworks include Scrum and Kanban."
    ]

    # Example reference answers
    references = [
        """
        An amateur built light airplane has crashed in the central New South Wales town of Mudgee .
        The plane was on its approach to land when the aircraft crashed into a paddock near the airport .
        The pilot died on the plane while his only passenger died at the crash site .
        """,
        "The French Revolution began in 1789.",
        "في أي عام بدأت الحرب العالمية الأولى؟",
        # """
        # عادت قضية اللجوء السوري إلى أوروبا إلى الواجهة بعد إعلان تركيا فتح حدودها أمام اللاجئين للهروب باتجاه اليونان. وأثارت مقاطع مصورة تظهر لاجئين يستقلون حافلات مجانية الجدل بين المغردين حول استخدام أردوغان "السوريين كسلاح في وجه أوروبا"، بين مؤيد لسياسته في الشرق الأوسط عامة وسوريا تحديداً ومعارض لها.
        # """,
        "Alexander Graham Bell is credited with inventing the telephone. While others were working on similar concepts, Bell received the first patent for the device.",
        # "في أي عام بدأت الحرب العالمية الأولى؟",
        "كيف يمكنني لعب اللعبة؟"
        # "يُنسب اختراع الهاتف إلى ألكسندر جراهام بيل.",
        # "Agile software development is an iterative and incremental approach to software development that emphasizes flexibility, collaboration, and customer satisfaction. Unlike traditional waterfall models, Agile methods embrace changing requirements throughout the project lifecycle.  Key characteristics include short development cycles (sprints), frequent releases, continuous feedback, self-organizing teams, and a focus on delivering working software as quickly as possible.  Popular Agile frameworks include Scrum and Kanban.",
        ""
    ]

    # Run batch evaluation
    generated_texts, avg_metrics = evaluate_batch(prompts, references)

    # Display average metrics
    print("\n" + "=" * 80)
    print("AVERAGE METRICS ACROSS ALL EXAMPLES")
    print("=" * 80)
    print(f"Average BLEU:        {avg_metrics['bleu']:.4f}")
    print(f"Average ROUGE-1:     {avg_metrics['rouge1']:.4f}")
    print(f"Average ROUGE-2:     {avg_metrics['rouge2']:.4f}")
    print(f"Average ROUGE-L:     {avg_metrics['rougeL']:.4f}")
    print(f"Average BERTScore:   {avg_metrics['bertscore_f1']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
