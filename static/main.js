// Image Caption Generator - Main JavaScript

class ImageCaptionGenerator {
    constructor() {
        this.currentImage = null;
        this.currentCaption = '';
        this.isGenerating = false;
        this.speechSynthesis = window.speechSynthesis;
        this.currentUtterance = null;
        
        this.initializeElements();
        this.bindEvents();
        this.setupSmoothScrolling();
    }

    initializeElements() {
        // Upload elements
        this.uploadArea = document.getElementById('uploadArea');
        this.uploadContent = document.getElementById('uploadContent');
        this.imageInput = document.getElementById('imageInput');
        
        // Display elements
        this.imageDisplay = document.getElementById('imageDisplay');
        this.uploadedImage = document.getElementById('uploadedImage');
        this.changeImageBtn = document.getElementById('changeImageBtn');
        
        // Control elements
        this.controlsSection = document.getElementById('controlsSection');
        this.languageSelect = document.getElementById('languageSelect');
        this.generateBtn = document.getElementById('generateBtn');
        
        // Caption elements
        this.captionSection = document.getElementById('captionSection');
        this.captionText = document.getElementById('captionText');
        this.languageBadge = document.getElementById('languageBadge');
        this.wordCount = document.getElementById('wordCount');
        this.copyBtn = document.getElementById('copyBtn');
        this.voiceBtn = document.getElementById('voiceBtn');
        
        // Toast element
        this.toast = document.getElementById('toast');
    }

    bindEvents() {
        // Upload events
        this.uploadArea.addEventListener('click', () => this.imageInput.click());
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        this.imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
        
        // Button events
        this.changeImageBtn.addEventListener('click', () => this.resetUpload());
        this.generateBtn.addEventListener('click', () => this.generateCaption());
        this.copyBtn.addEventListener('click', () => this.copyCaption());
        this.voiceBtn.addEventListener('click', () => this.toggleTextToSpeech());
        
        // Language change event
        this.languageSelect.addEventListener('change', () => this.onLanguageChange());
        
        // Prevent default drag behaviors on document
        document.addEventListener('dragover', (e) => e.preventDefault());
        document.addEventListener('drop', (e) => e.preventDefault());
    }

    setupSmoothScrolling() {
        // Smooth scrolling for navigation links
        document.querySelectorAll('.nav-link, .cta-button').forEach(link => {
            link.addEventListener('click', (e) => {
                const href = link.getAttribute('href');
                if (href && href.startsWith('#')) {
                    e.preventDefault();
                    const targetElement = document.querySelector(href);
                    if (targetElement) {
                        targetElement.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                }
            });
        });
    }

    // File upload handling
    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processImageFile(files[0]);
        }
    }

    handleImageUpload(e) {
        const file = e.target.files[0];
        if (file) {
            this.processImageFile(file);
        }
    }

    processImageFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showToast('Please select a valid image file.', 'error');
            return;
        }

        // Validate file size (10MB limit)
        const maxSize = 10 * 1024 * 1024; // 10MB in bytes
        if (file.size > maxSize) {
            this.showToast('File size must be less than 10MB.', 'error');
            return;
        }

        // Create file reader
        const reader = new FileReader();
        reader.onload = (e) => {
            this.displayUploadedImage(e.target.result, file.name);
        };
        reader.readAsDataURL(file);
    }

    displayUploadedImage(imageSrc, fileName) {
        this.currentImage = {
            src: imageSrc,
            name: fileName
        };

        // Update image display
        this.uploadedImage.src = imageSrc;
        this.uploadedImage.alt = `Uploaded image: ${fileName}`;
        
        // Show image display and controls
        this.imageDisplay.style.display = 'block';
        this.controlsSection.style.display = 'block';
        
        // Hide caption section if visible
        this.captionSection.style.display = 'none';
        
        // Scroll to image display
        setTimeout(() => {
            this.imageDisplay.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
        }, 300);

        this.showToast('Image uploaded successfully!');
    }

    resetUpload() {
        // Reset form and hide sections
        this.imageInput.value = '';
        this.currentImage = null;
        this.currentCaption = '';
        
        this.imageDisplay.style.display = 'none';
        this.controlsSection.style.display = 'none';
        this.captionSection.style.display = 'none';
        
        // Scroll back to upload area
        this.uploadArea.scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
        
        this.showToast('Upload reset. Please select a new image.');
    }

    onLanguageChange() {
        // If caption exists, show message about regenerating
        if (this.currentCaption) {
            this.showToast('Language changed. Click "Generate Caption" to get caption in the new language.');
        }
    }

    async generateCaption() {
        if (!this.currentImage) {
            this.showToast('Please upload an image first.', 'error');
            return;
        }

        if (this.isGenerating) {
            return;
        }

        this.isGenerating = true;
        this.updateGenerateButton(true);

        try {
            // Simulate AI processing time
            await this.simulateAIProcessing();
            
            // Generate caption based on selected language
            const selectedLanguage = this.languageSelect.value;
            const caption = this.generateMockCaption(selectedLanguage);
            
            this.displayCaption(caption, selectedLanguage);
            
        } catch (error) {
            this.showToast('Error generating caption. Please try again.', 'error');
        } finally {
            this.isGenerating = false;
            this.updateGenerateButton(false);
        }
    }

    simulateAIProcessing() {
        return new Promise(resolve => {
            // Simulate processing time between 2-4 seconds
            const processingTime = 2000 + Math.random() * 2000;
            setTimeout(resolve, processingTime);
        });
    }

    generateMockCaption(language) {
        // Mock captions in different languages
        const captions = {
            en: [
                "A beautiful landscape photograph showcasing rolling hills under a dramatic cloudy sky with vibrant colors.",
                "An artistic composition featuring architectural elements with interesting lighting and geometric patterns.",
                "A captivating portrait that demonstrates excellent use of natural lighting and compelling subject matter.",
                "A stunning nature scene with rich textures and harmonious color palette creating visual depth.",
                "An engaging street photography moment capturing authentic human interaction and urban atmosphere."
            ],
            es: [
                "Una hermosa fotografía de paisaje que muestra colinas ondulantes bajo un cielo dramático con colores vibrantes.",
                "Una composición artística que presenta elementos arquitectónicos con iluminación interesante y patrones geométricos.",
                "Un retrato cautivador que demuestra un excelente uso de la iluminación natural y un tema convincente.",
                "Una escena natural impresionante con texturas ricas y paleta de colores armoniosa creando profundidad visual.",
                "Un momento envolvente de fotografía callejera que captura la interacción humana auténtica y la atmósfera urbana."
            ],
            fr: [
                "Une belle photographie de paysage présentant des collines ondulantes sous un ciel dramatique aux couleurs vibrantes.",
                "Une composition artistique mettant en vedette des éléments architecturaux avec un éclairage intéressant et des motifs géométriques.",
                "Un portrait captivant qui démontre une excellente utilisation de l'éclairage naturel et un sujet convaincant.",
                "Une scène naturelle époustouflante avec des textures riches et une palette de couleurs harmonieuse créant une profondeur visuelle.",
                "Un moment engageant de photographie de rue capturant une interaction humaine authentique et une atmosphère urbaine."
            ],
            de: [
                "Eine schöne Landschaftsfotografie mit sanften Hügeln unter einem dramatischen bewölkten Himmel mit lebendigen Farben.",
                "Eine künstlerische Komposition mit architektonischen Elementen, interessanter Beleuchtung und geometrischen Mustern.",
                "Ein fesselndes Porträt, das eine ausgezeichnete Nutzung natürlichen Lichts und überzeugender Motive zeigt.",
                "Eine atemberaubende Naturszene mit reichen Texturen und harmonischer Farbpalette, die visuelle Tiefe schafft.",
                "Ein fesselnder Straßenfotografie-Moment, der authentische menschliche Interaktion und urbane Atmosphäre einfängt."
            ],
            it: [
                "Una bella fotografia paesaggistica che mostra colline ondulate sotto un cielo drammatico con colori vibranti.",
                "Una composizione artistica caratterizzata da elementi architettonici con illuminazione interessante e motivi geometrici.",
                "Un ritratto accattivante che dimostra un eccellente uso della luce naturale e soggetti convincenti.",
                "Una scena naturale mozzafiato con texture ricche e tavolozza di colori armoniosa che crea profondità visiva.",
                "Un momento coinvolgente di fotografia di strada che cattura interazione umana autentica e atmosfera urbana."
            ],
            pt: [
                "Uma bela fotografia de paisagem mostrando colinas ondulantes sob um céu dramático com cores vibrantes.",
                "Uma composição artística apresentando elementos arquitetônicos com iluminação interessante e padrões geométricos.",
                "Um retrato cativante que demonstra excelente uso de iluminação natural e assunto convincente.",
                "Uma cena natural deslumbrante com texturas ricas e paleta de cores harmoniosa criando profundidade visual.",
                "Um momento envolvente de fotografia de rua capturando interação humana autêntica e atmosfera urbana."
            ],
            zh: [
                "一张美丽的风景摄影作品，展现了戏剧性多云天空下起伏的山丘和充满活力的色彩。",
                "一个艺术构图，以有趣的照明和几何图案为特色的建筑元素。",
                "一张迷人的肖像，展现了对自然光线的出色运用和引人注目的主题。",
                "一个令人惊叹的自然场景，具有丰富的纹理和和谐的色彩调色板，创造出视觉深度。",
                "一个引人入胜的街头摄影瞬间，捕捉了真实的人类互动和城市氛围。"
            ],
            ja: [
                "劇的な曇り空の下の起伏のある丘と鮮やかな色彩を見せる美しい風景写真。",
                "興味深い照明と幾何学的パターンを持つ建築要素をフィーチャーした芸術的な構成。",
                "自然光の素晴らしい使用と説得力のある主題を示す魅力的な肖像画。",
                "豊かなテクスチャーと調和のとれたカラーパレットで視覚的深度を作り出す見事な自然シーン。",
                "本物の人間の相互作用と都市の雰囲気を捉えた魅力的なストリート写真の瞬間。"
            ],
            ko: [
                "극적인 구름 낀 하늘 아래 구불구불한 언덕과 생생한 색상을 보여주는 아름다운 풍경 사진.",
                "흥미로운 조명과 기하학적 패턴을 가진 건축적 요소를 특징으로 하는 예술적 구성.",
                "자연광의 뛰어난 사용과 설득력 있는 주제를 보여주는 매력적인 초상화.",
                "시각적 깊이를 만드는 풍부한 질감과 조화로운 색상 팔레트를 가진 놀라운 자연 장면.",
                "진정한 인간 상호작용과 도시 분위기를 포착한 매력적인 거리 사진 순간."
            ],
            ar: [
                "صورة طبيعية جميلة تظهر التلال المتموجة تحت سماء درامية مليئة بالغيوم مع ألوان نابضة بالحياة.",
                "تكوين فني يتميز بعناصر معمارية مع إضاءة مثيرة للاهتمام وأنماط هندسية.",
                "صورة شخصية آسرة تظهر الاستخدام الممتاز للإضاءة الطبيعية والموضوع المقنع.",
                "مشهد طبيعي مذهل مع قوام غني ولوحة ألوان متناغمة تخلق عمقاً بصرياً.",
                "لحظة مشاركة في التصوير الفوتوغرافي للشارع تلتقط التفاعل الإنساني الأصيل والأجواء الحضرية."
            ]
        };

        const languageCaptions = captions[language] || captions.en;
        const randomIndex = Math.floor(Math.random() * languageCaptions.length);
        return languageCaptions[randomIndex];
    }

    displayCaption(caption, language) {
        this.currentCaption = caption;
        
        // Update caption text with typewriter effect
        this.animateTextTyping(this.captionText, caption);
        
        // Update language badge
        const languageNames = {
            en: 'English',
            es: 'Spanish',
            fr: 'French',
            de: 'German',
            it: 'Italian',
            pt: 'Portuguese',
            zh: 'Chinese',
            ja: 'Japanese',
            ko: 'Korean',
            ar: 'Arabic'
        };
        this.languageBadge.textContent = languageNames[language] || 'English';
        
        // Update word count
        const wordCount = caption.split(' ').length;
        this.wordCount.textContent = `${wordCount} words`;
        
        // Show caption section
        this.captionSection.style.display = 'block';
        
        // Scroll to caption
        setTimeout(() => {
            this.captionSection.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
        }, 300);
        
        this.showToast('Caption generated successfully!');
    }

    animateTextTyping(element, text) {
        element.textContent = '';
        let index = 0;
        
        const typeWriter = () => {
            if (index < text.length) {
                element.textContent += text.charAt(index);
                index++;
                setTimeout(typeWriter, 30);
            }
        };
        
        typeWriter();
    }

    updateGenerateButton(isLoading) {
        const buttonText = this.generateBtn.querySelector('span');
        const buttonLoader = this.generateBtn.querySelector('.btn-loader');
        const buttonIcon = this.generateBtn.querySelector('i');

        if (isLoading) {
            this.generateBtn.disabled = true;
            buttonText.style.display = 'none';
            buttonIcon.style.display = 'none';
            buttonLoader.style.display = 'block';
        } else {
            this.generateBtn.disabled = false;
            buttonText.style.display = 'inline';
            buttonIcon.style.display = 'inline';
            buttonLoader.style.display = 'none';
        }
    }

    copyCaption() {
        if (!this.currentCaption) {
            this.showToast('No caption to copy.', 'error');
            return;
        }

        navigator.clipboard.writeText(this.currentCaption).then(() => {
            this.showToast('Caption copied to clipboard!');
            
            // Visual feedback
            this.copyBtn.classList.add('active');
            setTimeout(() => {
                this.copyBtn.classList.remove('active');
            }, 1000);
        }).catch(() => {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = this.currentCaption;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            this.showToast('Caption copied to clipboard!');
        });
    }

    toggleTextToSpeech() {
        if (!this.currentCaption) {
            this.showToast('No caption to read aloud.', 'error');
            return;
        }

        // Stop current speech if playing
        if (this.speechSynthesis.speaking) {
            this.speechSynthesis.cancel();
            this.voiceBtn.classList.remove('active');
            this.voiceBtn.querySelector('i').className = 'fas fa-volume-up';
            return;
        }

        // Create new utterance
        this.currentUtterance = new SpeechSynthesisUtterance(this.currentCaption);
        
        // Get selected language
        const selectedLanguage = this.languageSelect.value;
        const languageCodes = {
            en: 'en-US',
            es: 'es-ES',
            fr: 'fr-FR',
            de: 'de-DE',
            it: 'it-IT',
            pt: 'pt-PT',
            zh: 'zh-CN',
            ja: 'ja-JP',
            ko: 'ko-KR',
            ar: 'ar-SA'
        };
        
        this.currentUtterance.lang = languageCodes[selectedLanguage] || 'en-US';
        this.currentUtterance.rate = 0.9;
        this.currentUtterance.pitch = 1;
        this.currentUtterance.volume = 1;

        // Event listeners
        this.currentUtterance.onstart = () => {
            this.voiceBtn.classList.add('active');
            this.voiceBtn.querySelector('i').className = 'fas fa-stop';
            this.showToast('Reading caption aloud...');
        };

        this.currentUtterance.onend = () => {
            this.voiceBtn.classList.remove('active');
            this.voiceBtn.querySelector('i').className = 'fas fa-volume-up';
        };

        this.currentUtterance.onerror = () => {
            this.voiceBtn.classList.remove('active');
            this.voiceBtn.querySelector('i').className = 'fas fa-volume-up';
            this.showToast('Error reading caption aloud.', 'error');
        };

        // Start speaking
        this.speechSynthesis.speak(this.currentUtterance);
    }

    showToast(message, type = 'success') {
        this.toast.textContent = message;
        this.toast.className = `toast ${type}`;
        this.toast.classList.add('show');

        setTimeout(() => {
            this.toast.classList.remove('show');
        }, 3000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ImageCaptionGenerator();
});

// Add some interactive animations
document.addEventListener('DOMContentLoaded', () => {
    // Animate elements on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe feature cards and other elements
    document.querySelectorAll('.feature-card, .step, .caption-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});

// Add parallax effect to floating shapes
document.addEventListener('mousemove', (e) => {
    const shapes = document.querySelectorAll('.floating-shape');
    const x = e.clientX / window.innerWidth;
    const y = e.clientY / window.innerHeight;

    shapes.forEach((shape, index) => {
        const speed = (index + 1) * 0.5;
        const xPos = (x - 0.5) * speed * 50;
        const yPos = (y - 0.5) * speed * 50;
        
        shape.style.transform = `translate(${xPos}px, ${yPos}px)`;
    });
});