/**
 * VATGuard Static Site — minimal progressive enhancement
 * No frameworks, no build step. Works offline.
 */

(function () {
  'use strict';

  /* ── Smooth-scroll nav links ─────────────────────────────────────────── */
  document.querySelectorAll('a[href^="#"]').forEach(function (link) {
    link.addEventListener('click', function (e) {
      var target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  /* ── Active nav highlight on scroll ─────────────────────────────────── */
  var sections = document.querySelectorAll('section[id], header[id]');
  var navLinks = document.querySelectorAll('.nav-links a[href^="#"]');

  if (sections.length && navLinks.length) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          navLinks.forEach(function (link) {
            link.classList.toggle(
              'nav-active',
              link.getAttribute('href') === '#' + entry.target.id
            );
          });
        }
      });
    }, { rootMargin: '-20% 0px -70% 0px' });

    sections.forEach(function (s) { observer.observe(s); });
  }

  /* ── Fraud card entrance animation ──────────────────────────────────── */
  if ('IntersectionObserver' in window) {
    var cards = document.querySelectorAll('.fraud-card, .score-component, .task-card');
    var cardObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
          cardObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1 });

    cards.forEach(function (card) {
      card.style.opacity = '0';
      card.style.transform = 'translateY(16px)';
      card.style.transition = 'opacity .4s ease, transform .4s ease';
      cardObserver.observe(card);
    });
  }
})();
