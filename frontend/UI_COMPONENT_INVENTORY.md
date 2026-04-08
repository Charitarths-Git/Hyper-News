# HyperNews UI Component Inventory

## Purpose
- This file is the redesign inventory for the current HyperNews product surface.
- It lists the screens, major sections, shared components, controls, and UI states required to rebuild the app UI without relying on the current visual design.

## Global App Shell
- Brand header with logo, product name, signed-in state, and contextual navigation.
- Search input with suggestion dropdown and submit action.
- Mood selector pills.
- Explore-focus slider.
- Global primary action button style.
- Global secondary action button style.
- Toast / transient feedback surface.
- Loading skeleton cards.
- Empty-state blocks.
- Error-state alert blocks.

## Authentication Screens
- `Login` page
  - Headline and supporting copy.
  - Email input.
  - Password input.
  - Submit button.
  - Inline error message area.
  - Link to register page.
- `Register` page
  - Headline and supporting copy.
  - Display-name input.
  - Email input.
  - Password input.
  - Submit button.
  - Inline error message area.
  - Link to login page.

## Onboarding Screen
- Multi-step onboarding container.
- Step progress bar / progress indicator.
- Step title and description.
- Step 1: age bucket select, gender select.
- Step 2: occupation select, region select.
- Step 3: interest-notes textarea, top-categories input, affect-consent checkbox.
- Back button.
- Continue button.
- Final submit button.
- Validation / save error area.

## Feed Screen
- Context header.
- Guest-mode info banner.
- Explanation banner for recommendation mode.
- Recommendation grid.
- News card component.
- Load-more button / infinite-scroll sentinel.
- Feed loading state.
- Feed empty state.
- Feed refresh state.

## Search Screen
- Shared context header.
- Search query field and suggestion list.
- Search results grid.
- Search explanation banner.
- Search loading state.
- Search empty state.
- Search prompt state when no query is active.

## Dashboard Screen
- Back-to-feed link.
- Link to profile/settings.
- Overview stat cards:
  - Articles read.
  - Positive interactions.
  - Current mood.
  - Average dwell time.
- Interest graph panel.
- Profile snapshot panel.
- Knowledge graph panel.
- Recent clicks section.
- Recent negative signals section.
- Recent searches section.
- Recent feedback section.
- Recent entities section.
- Recent sources section.

## Profile / Settings Screen
- Back-to-feed link.
- Link to dashboard.
- Overview stat cards:
  - Positive interactions.
  - Bio encoder status.
  - Onboarding status.
- Editable profile form:
  - Display name.
  - Age bucket.
  - Gender.
  - Occupation.
  - Region.
  - Top categories.
  - Interest notes.
  - Affect consent.
- Save confirmation area.
- Save error area.
- Save button.

## Shared Content Components
- `NewsCard`
  - Category badge.
  - Candidate-source label.
  - Score label.
  - Headline.
  - Abstract / summary text.
  - Reason chips.
  - Matched-entity chips.
  - Score bar.
  - Feedback action row.
- `ExplanationBanner`
  - Mode label.
  - Explanation text.
- `InterestChart`
  - Horizontal bar chart.
  - Tooltip.
  - Empty state when no interests exist.
- `KnowledgeGraphPanel`
  - Graph canvas.
  - Expand / collapse toggle.
  - Legend.
  - Top-entity chips.

## Buttons And Controls
- Primary CTA button.
- Secondary button.
- Ghost link button.
- Mood pill.
- Range slider.
- Search submit button.
- Refresh button.
- Reset-profile button.
- Sign-in button.
- Sign-out button.
- Register button.
- Back button.
- Continue button.
- Final onboarding submit button.
- Feedback buttons:
  - Read.
  - Save.
  - More like this.
  - Skip.
  - Not interested.
  - Less from source.

## Form Elements
- Single-line text input.
- Email input.
- Password input.
- Category tag input.
- Select dropdown.
- Multiline textarea.
- Checkbox.
- Inline helper text.
- Inline error text.

## Page States To Design
- Guest vs authenticated header states.
- Authenticated but onboarding-incomplete redirect path.
- Feed loading, refresh, empty, and exhausted states.
- Search idle, loading, empty, and exhausted states.
- Dashboard with no interaction history yet.
- Profile save success and failure states.
- Onboarding submit success and failure states.
- Graph unavailable state.

## Responsive Requirements
- Header actions wrap cleanly on tablet/mobile.
- Auth and onboarding forms stay readable on narrow screens.
- Feed and search cards collapse to one column on mobile.
- Dashboard stats and panels reflow into stacked sections.
- Profile form uses single-column layout on smaller screens.
