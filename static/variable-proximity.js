class VariableProximity {
  constructor(element, options) {
    options = options || {};
    this.element = element;
    this.radius       = options.radius       !== undefined ? options.radius       : 100;
    this.falloff      = options.falloff      !== undefined ? options.falloff      : 'gaussian';
    this.fromSettings = options.fromSettings !== undefined ? options.fromSettings : "'wght' 400, 'opsz' 9";
    this.toSettings   = options.toSettings   !== undefined ? options.toSettings   : "'wght' 1000, 'opsz' 40";

    this.mouse     = { x: -9999, y: -9999 };
    this.lastMouse = { x: -9998, y: -9998 };
    this.rafId     = null;
    this.letters   = [];
    this._rects    = [];   // cached page-relative letter centers

    this._parseSettings();
    this._buildLetters();
    this._cacheRects();
    this._bindEvents();
    this._loop();
  }

  _parseSettings() {
    var parseStr = function(str) {
      return str.split(',').map(function(part) {
        var m = part.trim().match(/'([^']+)'\s+([\d.+eE-]+)/);
        return m ? { axis: m[1], value: parseFloat(m[2]) } : null;
      }).filter(Boolean);
    };
    this.fromAxes = parseStr(this.fromSettings);
    this.toAxes   = parseStr(this.toSettings);
  }

  _buildLetters() {
    var self = this;
    Array.from(this.element.childNodes).forEach(function(node) {
      if (node.nodeType !== Node.ELEMENT_NODE || node.tagName !== 'SPAN') return;

      var lineSpan   = node;
      var text       = lineSpan.textContent;
      var isGradient = lineSpan.classList.contains('gradient-text');

      lineSpan.textContent = '';

      Array.from(text).forEach(function(char) {
        var span = document.createElement('span');
        span.textContent = char;
        span.style.display = 'inline';

        if (isGradient) {
          span.style.background = 'linear-gradient(90deg, #fff 0%, #3498db 60%, #9b59b6 100%)';
          span.style.webkitBackgroundClip = 'text';
          span.style.backgroundClip = 'text';
        }

        lineSpan.appendChild(span);
        self.letters.push(span);
      });
    });
  }

  // Cache letter centers as page-relative coords (scrollY-independent recalc on resize only)
  _cacheRects() {
    var scrollX = window.pageXOffset || 0;
    var scrollY = window.pageYOffset || 0;
    this._rects = this.letters.map(function(letter) {
      var r = letter.getBoundingClientRect();
      return {
        cx: r.left + r.width  / 2 + scrollX,
        cy: r.top  + r.height / 2 + scrollY
      };
    });
  }

  _falloff(distance) {
    var radius = this.radius;
    if (this.falloff === 'linear') {
      var norm = 1 - distance / radius;
      return norm < 0 ? 0 : norm > 1 ? 1 : norm;
    }
    if (this.falloff === 'exponential') {
      var norm = 1 - distance / radius;
      norm = norm < 0 ? 0 : norm > 1 ? 1 : norm;
      return norm * norm;
    }
    // gaussian
    var sigma = radius / 2;
    var ratio = distance / sigma;
    return Math.exp(-(ratio * ratio) / 2);
  }

  _lerp(a, b, t) { return a + (b - a) * t; }

  _loop() {
    var self = this;
    self.rafId = requestAnimationFrame(function() { self._loop(); });

    if (self.mouse.x === self.lastMouse.x && self.mouse.y === self.lastMouse.y) return;
    self.lastMouse.x = self.mouse.x;
    self.lastMouse.y = self.mouse.y;

    var mx      = self.mouse.x;
    var my      = self.mouse.y;
    var scrollX = window.pageXOffset || 0;
    var scrollY = window.pageYOffset || 0;

    self.letters.forEach(function(letter, i) {
      var rect = self._rects[i];
      var dx   = mx - (rect.cx - scrollX);
      var dy   = my - (rect.cy - scrollY);
      var dist = Math.sqrt(dx * dx + dy * dy);
      var t    = self._falloff(dist);

      var parts = self.fromAxes.map(function(from, j) {
        var val = self._lerp(from.value, self.toAxes[j].value, t);
        return "'" + from.axis + "' " + val.toFixed(2);
      });

      letter.style.fontVariationSettings = parts.join(', ');
    });
  }

  _bindEvents() {
    var self = this;
    self._onMouseMove = function(e) {
      self.mouse.x = e.clientX;
      self.mouse.y = e.clientY;
    };
    self._onTouchMove = function(e) {
      if (!e.touches.length) return;
      self.mouse.x = e.touches[0].clientX;
      self.mouse.y = e.touches[0].clientY;
    };
    self._onResize = function() { self._cacheRects(); };

    window.addEventListener('mousemove', self._onMouseMove);
    window.addEventListener('touchmove', self._onTouchMove, { passive: true });
    window.addEventListener('resize',    self._onResize);
  }

  destroy() {
    cancelAnimationFrame(this.rafId);
    window.removeEventListener('mousemove', this._onMouseMove);
    window.removeEventListener('touchmove', this._onTouchMove);
    window.removeEventListener('resize',    this._onResize);
    this.letters = [];
    this._rects  = [];
  }
}
