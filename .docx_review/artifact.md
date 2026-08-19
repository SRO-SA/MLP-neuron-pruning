# Version 3 template execution contract

## Reference

- Source: `C:\Users\srosa\Research\MLP Neuron Pruning\bound_guided_swiglu_moe_pruning_research_draft_v2.docx`
- SHA-256: `C853B6FC1FD6ECC4BCA201586282B773C5716F1E15E142321CB878A4E9D09D76`
- Package parts: 21
- Sections: 1
- Page count: unresolved; the source `docProps/app.xml` value is stale and LibreOffice is unavailable for rendering.
- Structural evidence: section/style/heading/image/field/footnote audits run on 2026-08-19.
- Render evidence: unavailable because the bundled renderer could not find LibreOffice/soffice.

## Page system

- US Letter portrait, 8.5 x 11 inches.
- One section, NEW_PAGE start type.
- Margins: 0.75 inches on all sides.
- Header linked; blank header.
- Footer not linked and contains the working-draft title.
- No different-first-page or odd/even header mode.

## Typography

- Normal: Times New Roman, 10.5 pt.
- Title: Times New Roman, 20 pt, color `17365D`, single line spacing, 15 pt after.
- Heading 1: Times New Roman, 15 pt, bold, color `365F91`, 24 pt before.
- Heading 2: Times New Roman, 13 pt, bold, color `4F81BD`, 10 pt before.
- Heading 3: Times New Roman, 11.5 pt, bold, color `4F81BD`, 10 pt before.
- Caption: Times New Roman, 9 pt, bold italic, color `4F81BD`, single spacing.
- Existing document uses direct run formatting extensively; new material should reuse named styles and clone source table formatting.

## Lists, tables, and figures

- Existing bullets use the real `List Bullet` style and numbering definitions; preserve `word/numbering.xml`.
- Existing tables use `Table Grid`; retain existing tables verbatim.
- New result tables may clone the source `Table Grid` treatment, use repeated header styling, and fit the 7-inch text width.
- Existing document contains 12 tables and three inline figures, each approximately 6.20 x 3.82 inches. Preserve all three images and their relationships.

## Components

- Title block: title, working-draft subtitle, author, date.
- Numbered Heading 1 sections, Heading 2/3 subsections, blue italic captions, grid tables, three benchmark figures.
- Footer: `Bound-Guided Structured SwiGLU and MoE Expert-Channel Pruning — Working Draft`.
- No Word fields, footnotes, or endnotes.

## Content flow and slot map

- Title/subtitle/date: rewrite for Version 3 while preserving the source title-block pattern.
- Abstract and keywords: rewrite to summarize both the historical residual work and the ellipsoid expansion.
- Introduction/contributions: preserve prior dense/MoE/residual contributions and add the exact ellipsoid bound, allocation/ranking separation, paired evaluation, and tightness audit.
- Related work: preserve all prior discussion and references; add a clearly marked note that the 2026 closest-work review remains to be finalized.
- Method: preserve Sections 3.1-3.5; expand Section 3.2 from the old spherical proxy to the exact RMSNorm ellipsoid derivation and add allocation/ranking and certification-scope subsections.
- Implementation/reproducibility: preserve and extend with fresh-model validation, independent allocation/ranking controls, paired bootstrap comparison, and tightness instrumentation.
- Experimental setup: preserve prior setup and add a clearly labeled Version 3 controlled protocol.
- Dense results: preserve all tables and prose; label as original implementation evidence.
- Historical MoE results: preserve all tables, figures, residual results, and prose; label as Version 2 evidence using n_eval=512.
- New ellipsoid results: insert new sections/tables for the 2% diagnostic, 4% allocation/ranking study, 6% controlled study, paired confidence intervals, and bound tightness.
- Achievements/positioning/limitations/next experiments/conclusion: preserve prior information and add Version 3 implications and remaining gates.
- References and appendices: preserve all existing references and appendices; add internal artifacts and a Version 3 mathematical appendix. Do not delete historical commands or observations.

## Package preservation

- Preserve unchanged: styles, stylesWithEffects, numbering, font table, theme, customXml, all three media images, existing relationships, and footer styling.
- Editable: `word/document.xml`, core properties, footer text, and settings only if needed for update-on-open.
- Source document must remain byte-for-byte unchanged.

## Fidelity gates

- Final remains recognizably derived from the source: same page geometry, fonts, heading colors, captions, tables, images, footer pattern, and section hierarchy.
- All Version 2 tables, figures, references, limitations, and appendices remain present.
- New equations are readable plain mathematical text using Cambria Math where practical.
- No unsupported claim that p95 aggregation is a uniform all-expert certificate.
- New and old evaluation protocols are explicitly separated; raw PPL values from n_eval=512 and n_eval=1024 are not presented as directly interchangeable.
- Structural audits must confirm one section, three preserved images, no lost tables, and no lost package relationships.
- Visual render QA is required if a renderer becomes available; otherwise perform structural and content audits and disclose the limitation.
